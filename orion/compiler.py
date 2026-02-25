"""Compiler: traces, fits, and compiles a network for FHE inference.

Requires Go backend (Lattigo). No cryptographic keys needed.
Produces a CompiledModel containing pre-encoded LinearTransform blobs,
polynomial coefficients, module metadata, and a KeyManifest.
"""

import sys
import time
import math
import types

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from orion.params import CKKSParams, CompilerConfig
from orion.compiled_model import CompiledModel, KeyManifest
from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.core.compiler_backend import (
    CompilerBackend,
    NewParameters,
    NewEncoder,
    PolynomialGenerator,
    TransformEncoder,
)
from orion.core.tracer import StatsTracker, OrionTracer
from orion.core.fuser import Fuser
from orion.core.network_dag import NetworkDAG
from orion.core.auto_bootstrap import BootstrapSolver, BootstrapPlacer


class Compiler:
    """Compiles a neural network for FHE inference.

    Usage:
        compiler = Compiler(net, params)
        compiler.fit(dataloader)
        compiled = compiler.compile()  # -> CompiledModel
    """

    def __init__(
        self,
        net: Module,
        params: CKKSParams,
        config: CompilerConfig | None = None,
    ):
        self.net = net
        self.ckks_params = params
        self.config = config or CompilerConfig()

        # Build legacy NewParameters from v2 dataclasses
        self.params = NewParameters.from_ckks_params(
            params, self.config
        )

        # Initialize Go backend (no keys)
        self.backend = CompilerBackend()
        self.backend.setup_bindings(self.params)

        # Build compile-time wrappers
        self._encoder = NewEncoder(self)
        self._poly_evaluator = PolynomialGenerator(self.backend)
        self._lt_evaluator = TransformEncoder(self.backend, self.params)

        # Will be set by fit()
        self._traced = None
        self._margin = self.config.margin

        # Build the context namespace that modules will receive
        self._context = self._build_context()

    # -- Properties expected by NewEncoder and other wrappers --

    @property
    def encoder(self):
        return self._encoder

    @property
    def lt_evaluator(self):
        return self._lt_evaluator

    @property
    def poly_evaluator(self):
        return self._poly_evaluator

    @property
    def margin(self):
        return self._margin

    def _build_context(self):
        """Build a namespace object for module.compile(context) and module.fit(context)."""
        ctx = types.SimpleNamespace()
        ctx.backend = self.backend
        ctx.params = self.params
        ctx.encoder = self._encoder
        ctx.lt_evaluator = self._lt_evaluator
        ctx.poly_evaluator = self._poly_evaluator
        ctx.margin = self._margin
        ctx.config = self.config
        # Not available in compile mode
        ctx.evaluator = None
        ctx.encryptor = None
        ctx.bootstrapper = None
        return ctx

    def fit(self, input_data, batch_size=128):
        """Run cleartext forward passes to collect per-layer statistics.

        Traces the model, records min/max ranges, and fits polynomial
        approximations for activation functions.
        """
        net = self.net

        tracer = OrionTracer()
        self._traced = tracer.trace_model(net)

        stats_tracker = StatsTracker(self._traced)

        # Device detection
        param = next(iter(net.parameters()), None)
        device = param.device if param is not None else torch.device("cpu")

        print("\n[1/5] Finding per-layer input/output ranges and shapes...",
              flush=True)
        start = time.time()
        if isinstance(input_data, DataLoader):
            user_batch_size = input_data.batch_size
            if user_batch_size is not None and batch_size > user_batch_size:
                dataset = input_data.dataset
                shuffle = (input_data.sampler is None or
                           isinstance(input_data.sampler, RandomSampler))
                input_data = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=input_data.num_workers,
                    pin_memory=input_data.pin_memory,
                    drop_last=input_data.drop_last,
                )

            for batch in tqdm(input_data, desc="Processing input data",
                              unit="batch", leave=True):
                stats_tracker.propagate(batch[0].to(device))

            stats_tracker.update_batch_size(user_batch_size)

        elif isinstance(input_data, torch.Tensor):
            stats_tracker.propagate(input_data.to(device))
        else:
            raise ValueError(
                "Input data must be a torch.Tensor or DataLoader, but "
                f"received {type(input_data)}."
            )

        # Fit polynomial activations
        print("\n[2/5] Fitting polynomials... ", end="", flush=True)
        start = time.time()
        for module in net.modules():
            if hasattr(module, "fit") and callable(module.fit):
                module.fit(self._context)
        print(f"done! [{time.time()-start:.3f} secs.]")

    def compile(self) -> CompiledModel:
        """Compile the fitted network into a CompiledModel.

        Returns a CompiledModel containing all artifacts needed by
        Client (for key generation) and Evaluator (for inference).
        """
        if self._traced is None:
            raise ValueError(
                "Network has not been fit yet! Call compiler.fit(data) "
                "before compiler.compile()."
            )

        net = self.net

        # Build DAG
        network_dag = NetworkDAG(self._traced)
        network_dag.build_dag()

        # Initialize Orion params (clone weights/biases)
        for module in net.modules():
            if hasattr(module, "init_orion_params") and callable(
                module.init_orion_params
            ):
                module.init_orion_params()

        # Resolve pooling kernels
        for module in net.modules():
            if hasattr(module, "update_params") and callable(
                module.update_params
            ):
                module.update_params()

        # Set scheme ref on modules for backward compat with packing.py
        for module in net.modules():
            if hasattr(module, "he_mode"):
                module.scheme = self

        # Fuse modules
        if self.config.fuse_modules:
            fuser = Fuser(network_dag)
            fuser.fuse_modules()
            network_dag.remove_fused_batchnorms()

        # Force last linear to square embedding
        topo_sort = list(network_dag.topological_sort())
        last_linear = None
        for node in reversed(topo_sort):
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                last_linear = node
                break

        # Generate diagonals
        print("\n[3/5] Generating matrix diagonals...", flush=True)
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                print(f"\nPacking {node}:")
                module.generate_diagonals(last=(node == last_linear))

        # Find residual connections
        network_dag.find_residuals()

        # Solve bootstrap placement
        print("\n[4/5] Running bootstrap placement... ", end="", flush=True)
        start = time.time()
        l_eff = len(self.params.get_logq()) - 1
        btp_solver = BootstrapSolver(
            net, network_dag, l_eff=l_eff, context=self._context
        )
        input_level, num_bootstraps, bootstrapper_slots = btp_solver.solve()
        print(f"done! [{time.time()-start:.3f} secs.]", flush=True)
        print(
            f"├── Network requires {num_bootstraps} bootstrap "
            f"{'operation' if num_bootstraps == 1 else 'operations'}."
        )

        if bootstrapper_slots:
            slots_str = ", ".join(
                [str(int(math.log2(slot))) for slot in bootstrapper_slots]
            )
            print(
                f"├── [compiler] Recorded bootstrap slots for logslots = "
                f"{slots_str} (no key generation in compiler mode)"
            )

        btp_placer = BootstrapPlacer(net, network_dag, self._context)
        btp_placer.place_bootstraps()

        # Compile all modules and collect artifacts
        print("\n[5/5] Compiling network layers...", flush=True)
        blobs = []
        module_metadata = {}
        topology = []

        for node in topo_sort:
            node_attrs = network_dag.nodes[node]
            module = node_attrs["module"]

            if isinstance(module, LinearTransform):
                # Compile (generates transforms + encodes diags)
                module.compile(self._context)

                # Serialize each transform blob
                transform_blobs = {}
                for (row, col), tid in module.transform_ids.items():
                    blob_data = self.backend.SerializeLinearTransform(tid)
                    transform_blobs[f"{row},{col}"] = len(blobs)
                    blobs.append(bytes(blob_data))

                # Free compile-time LT handles (serialized into blobs, no longer needed)
                self._lt_evaluator.delete_transforms(module.transform_ids)

                # Serialize bias as raw float64
                bias_vec = module.on_bias_ptxt
                bias_decoded = self._encoder.decode(bias_vec)
                bias_bytes = bias_decoded.numpy().astype("float64").tobytes()
                bias_blob_idx = len(blobs)
                blobs.append(bias_bytes)

                meta = {
                    "type": type(module).__name__,
                    "level": module.level,
                    "depth": module.depth,
                    "fused": module.fused,
                    "bsgs_ratio": module.bsgs_ratio,
                    "output_rotations": module.output_rotations,
                    "transform_blobs": transform_blobs,
                    "bias_blob": bias_blob_idx,
                    "fhe_input_shape": list(module.fhe_input_shape)
                    if hasattr(module, "fhe_input_shape")
                    else None,
                    "fhe_output_shape": list(module.fhe_output_shape)
                    if hasattr(module, "fhe_output_shape")
                    else None,
                    "output_shape": list(module.output_shape)
                    if hasattr(module, "output_shape")
                    else None,
                    "input_shape": list(module.input_shape)
                    if hasattr(module, "input_shape")
                    else None,
                }
                module_metadata[node] = meta
                topology.append(node)

            elif isinstance(module, Module) and self._has_own_compile(module):
                print(f"├── {node} @ level={module.level}", flush=True)
                module.compile(self._context)

                meta = self._extract_module_metadata(node, module)
                if meta is not None:
                    module_metadata[node] = meta
                    topology.append(node)

            elif isinstance(module, Module):
                # Modules without compile() (Quad, Flatten, Add, Mult)
                # Still record their level/depth for Evaluator reconstruction
                meta = self._extract_module_metadata(node, module)
                if meta is not None:
                    module_metadata[node] = meta
                    topology.append(node)

        # Record modules not in the DAG (e.g. fused BatchNorms removed by Fuser)
        for mod_name, module in net.named_modules():
            if mod_name not in module_metadata and isinstance(module, Module):
                meta = self._extract_module_metadata(mod_name, module)
                if meta is not None:
                    module_metadata[mod_name] = meta

        # Extract bootstrap hook metadata
        self._extract_bootstrap_metadata(
            net, network_dag, module_metadata, topology, blobs
        )

        # Collect all Galois elements
        galois_elements = set()
        galois_elements.update(self._lt_evaluator.required_galois_elements)
        galois_elements.update(self._get_po2_galois_elements())

        # Hybrid output rotations
        slots = self.params.get_slots()
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                for i in range(1, module.output_rotations + 1):
                    rotation = slots // (2**i)
                    gal_el = self.backend.GetGaloisElement(rotation)
                    galois_elements.add(gal_el)

        manifest = KeyManifest(
            galois_elements=frozenset(galois_elements),
            bootstrap_slots=tuple(sorted(bootstrapper_slots))
            if bootstrapper_slots
            else (),
            boot_logp=tuple(self.params.get_boot_logp())
            if bootstrapper_slots
            else None,
            needs_rlk=True,
        )

        compiled = CompiledModel(
            params=self.ckks_params,
            config=self.config,
            manifest=manifest,
            input_level=input_level,
            module_metadata=module_metadata,
            topology=topology,
            blobs=blobs,
        )

        return compiled

    def _extract_module_metadata(self, name, module):
        """Extract metadata dict from a compiled module."""
        from orion.nn.activation import (
            Activation,
            Chebyshev,
            Quad,
            ReLU,
        )
        from orion.nn.normalization import BatchNorm1d, BatchNorm2d
        from orion.nn.operations import Bootstrap

        if isinstance(module, Chebyshev):
            return {
                "type": "Chebyshev",
                "level": module.level,
                "depth": module.depth,
                "fused": module.fused,
                "coeffs": module.coeffs,
                "prescale": module.prescale,
                "constant": module.constant,
                "output_scale": module.output_scale,
                "input_min": float(module.input_min)
                if hasattr(module, "input_min")
                else None,
                "input_max": float(module.input_max)
                if hasattr(module, "input_max")
                else None,
                "degree": module.degree,
                "fhe_input_shape": list(module.fhe_input_shape)
                if hasattr(module, "fhe_input_shape")
                else None,
                "fhe_output_shape": list(module.fhe_output_shape)
                if hasattr(module, "fhe_output_shape")
                else None,
            }

        if isinstance(module, Activation):
            return {
                "type": "Activation",
                "level": module.level,
                "depth": module.depth,
                "fused": module.fused,
                "coeffs": list(module.coeffs)
                if hasattr(module, "coeffs")
                else None,
                "output_scale": module.output_scale,
                "fhe_input_shape": list(module.fhe_input_shape)
                if hasattr(module, "fhe_input_shape")
                else None,
                "fhe_output_shape": list(module.fhe_output_shape)
                if hasattr(module, "fhe_output_shape")
                else None,
            }

        if isinstance(module, Quad):
            return {
                "type": "Quad",
                "level": module.level,
                "depth": module.depth,
                "fhe_input_shape": list(module.fhe_input_shape)
                if hasattr(module, "fhe_input_shape")
                else None,
                "fhe_output_shape": list(module.fhe_output_shape)
                if hasattr(module, "fhe_output_shape")
                else None,
            }

        if isinstance(module, (BatchNorm1d, BatchNorm2d)):
            return {
                "type": type(module).__name__,
                "level": module.level,
                "depth": module.depth,
                "fused": module.fused,
                "fhe_input_shape": list(module.fhe_input_shape)
                if hasattr(module, "fhe_input_shape")
                else None,
                "fhe_output_shape": list(module.fhe_output_shape)
                if hasattr(module, "fhe_output_shape")
                else None,
            }

        if isinstance(module, Bootstrap):
            return {
                "type": "Bootstrap",
                "input_level": module.input_level,
                "input_min": float(module.input_min),
                "input_max": float(module.input_max),
                "prescale": module.prescale,
                "postscale": module.postscale,
                "constant": module.constant,
                "fhe_input_shape": list(module.fhe_input_shape)
                if hasattr(module, "fhe_input_shape")
                else None,
            }

        if isinstance(module, ReLU):
            return {
                "type": "ReLU",
                "level": module.level,
                "depth": module.depth,
                "prescale": module.prescale,
                "postscale": module.postscale,
                "fhe_input_shape": list(module.fhe_input_shape)
                if hasattr(module, "fhe_input_shape")
                else None,
                "fhe_output_shape": list(module.fhe_output_shape)
                if hasattr(module, "fhe_output_shape")
                else None,
            }

        # Generic fallback
        return {
            "type": type(module).__name__,
            "level": module.level,
            "depth": module.depth,
            "fused": getattr(module, "fused", False),
        }

    def _extract_bootstrap_metadata(
        self, net, network_dag, module_metadata, topology, blobs
    ):
        """Extract bootstrap hooks into module_metadata."""
        from orion.nn.operations import Bootstrap

        boot_idx = 0
        for node in network_dag.topological_sort():
            module = network_dag.nodes[node]["module"]
            if hasattr(module, "bootstrapper") and isinstance(
                module.bootstrapper, Bootstrap
            ):
                bootstrapper = module.bootstrapper
                boot_name = f"__bootstrap_{boot_idx}"
                elements = bootstrapper.fhe_input_shape.numel()
                slots = 2 ** math.ceil(math.log2(elements))

                module_metadata[boot_name] = {
                    "type": "Bootstrap",
                    "hook_target": node,
                    "input_level": bootstrapper.input_level,
                    "input_min": float(bootstrapper.input_min),
                    "input_max": float(bootstrapper.input_max),
                    "prescale": bootstrapper.prescale,
                    "postscale": bootstrapper.postscale,
                    "constant": bootstrapper.constant,
                    "fhe_input_shape": list(bootstrapper.fhe_input_shape),
                    "slots": slots,
                }
                topology.append(boot_name)
                boot_idx += 1

    def _get_po2_galois_elements(self):
        """Get Galois elements for power-of-2 rotations."""
        elements = set()
        max_slots = self.backend.GetMaxSlots()
        i = 1
        while i < max_slots:
            gal_el = self.backend.GetGaloisElement(i)
            elements.add(gal_el)
            i *= 2
        return elements

    @staticmethod
    def _has_own_compile(module):
        """Check if module defines its own compile() (not from nn.Module)."""
        for cls in type(module).__mro__:
            if cls is Module or cls is object:
                return False
            if "compile" in cls.__dict__:
                return True
        return False

    def __del__(self):
        if hasattr(self, "backend") and self.backend:
            if "sys" in globals() and sys.modules:
                try:
                    self.backend.DeleteScheme()
                except Exception:
                    pass
