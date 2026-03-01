"""Compiler: traces, fits, and compiles a network for FHE inference.

Requires Go backend (Lattigo) for fit(). compile() is Go-free.
Produces a CompiledModel containing raw diagonal blobs, computation
graph (nodes + edges), polynomial coefficients, and a KeyManifest.
"""

import time
import math
import types

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from orion.params import CKKSParams, CostProfile, CompilerConfig
from orion.compiled_model import (
    CompiledModel,
    KeyManifest,
    GraphNode,
    GraphEdge,
    Graph,
    pack_raw_diagonals,
    pack_raw_bias,
)
from orion.nn.module import Module
from orion.nn.linear import LinearTransform, Conv2d
from orion.nn.activation import Activation, Chebyshev, Quad
from orion.nn.operations import Bootstrap, Add, Mult
from orion.nn.reshape import Flatten
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
from orion.core.galois import (
    compute_galois_elements_for_linear_transform,
    galois_element,
    nth_root_for_ring,
)
from orion.core import packing


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
        """Build a namespace object for module.fit(context)."""
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

        This method has ZERO Go/Lattigo dependency — only fit() needs Go.
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

        # -- v2: Build graph nodes, edges, blobs (no module.compile() calls) --
        print("\n[5/5] Building computation graph...", flush=True)

        max_slots = self.params.get_slots()
        nth_root = nth_root_for_ring(
            self.ckks_params.logn, self.ckks_params.ring_type
        )
        slots = max_slots

        blobs = []
        graph_nodes = []
        galois_elements = set()

        # Re-read topo_sort after bootstrap insertion (new nodes added)
        topo_sort = list(network_dag.topological_sort())

        for node in topo_sort:
            node_attrs = network_dag.nodes[node]
            module = node_attrs["module"]

            if module is None:
                # Skip auxiliary nodes (fork/join have module=None)
                continue

            graph_node = self._build_graph_node(
                node, module, blobs, max_slots, slots, nth_root,
                galois_elements
            )
            if graph_node is not None:
                graph_nodes.append(graph_node)

        # Extract edges, filtering out fork/join auxiliary nodes
        graph_edges = self._extract_edges(network_dag)

        # Determine graph input/output
        dst_names = {e.dst for e in graph_edges}
        src_names = {e.src for e in graph_edges}
        # Input: first node with no incoming edges
        graph_input = None
        for n in graph_nodes:
            if n.name not in dst_names:
                graph_input = n.name
                break
        # Output: last node with no outgoing edges
        graph_output = None
        for n in reversed(graph_nodes):
            if n.name not in src_names:
                graph_output = n.name
                break

        if graph_input is None:
            raise ValueError(
                "Could not determine graph input: no node without incoming edges"
            )
        if graph_output is None:
            raise ValueError(
                "Could not determine graph output: no node without outgoing edges"
            )

        graph = Graph(
            input=graph_input,
            output=graph_output,
            nodes=graph_nodes,
            edges=graph_edges,
        )

        # Power-of-2 rotation Galois elements (pure Python)
        i = 1
        while i < slots:
            galois_elements.add(galois_element(i, nth_root))
            i *= 2

        # Output rotation Galois elements
        for n in graph_nodes:
            if n.op == "linear_transform":
                out_rot = n.config.get("output_rotations", 0)
                for j in range(1, out_rot + 1):
                    rotation = slots // (2**j)
                    galois_elements.add(galois_element(rotation, nth_root))

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

        cost = CostProfile(
            bootstrap_count=num_bootstraps,
            galois_key_count=len(manifest.galois_elements),
            bootstrap_key_count=len(manifest.bootstrap_slots),
        )

        compiled = CompiledModel(
            params=self.ckks_params,
            config=self.config,
            manifest=manifest,
            input_level=input_level,
            cost=cost,
            graph=graph,
            blobs=blobs,
        )

        return compiled

    def _build_graph_node(
        self, node_name, module, blobs, max_slots, slots, nth_root,
        galois_elements
    ):
        """Build a GraphNode from a DAG node + module."""

        op = self._module_to_op(module)
        if op is None:
            return None

        level = getattr(module, "level", 0) or 0
        depth = getattr(module, "depth", 0) or 0
        config = {}
        shape = None
        blob_refs = None

        if isinstance(module, LinearTransform):
            # Raw diagonals -> blobs (no Go calls)
            blob_refs = {}
            for (row, col), diag_dict in module.diagonals.items():
                blob_data = pack_raw_diagonals(diag_dict, max_slots)
                blob_refs[f"diag_{row}_{col}"] = len(blobs)
                blobs.append(blob_data)

            # Raw bias -> blob (different constructors for Linear vs Conv2d)
            if isinstance(module, Conv2d):
                bias_vec = packing.construct_conv2d_bias(module)
            else:
                bias_vec = packing.construct_linear_bias(module)
            bias_data = pack_raw_bias(bias_vec.tolist(), max_slots)
            blob_refs["bias"] = len(blobs)
            blobs.append(bias_data)

            config = {
                "bsgs_ratio": module.bsgs_ratio,
                "output_rotations": module.output_rotations,
            }

            shape = {}
            if hasattr(module, "fhe_input_shape") and module.fhe_input_shape is not None:
                shape["fhe_input"] = list(module.fhe_input_shape)
            if hasattr(module, "fhe_output_shape") and module.fhe_output_shape is not None:
                shape["fhe_output"] = list(module.fhe_output_shape)
            if hasattr(module, "input_shape") and module.input_shape is not None:
                shape["input"] = list(module.input_shape)
            if hasattr(module, "output_shape") and module.output_shape is not None:
                shape["output"] = list(module.output_shape)

            # Compute Galois elements for this LT (pure Python)
            diag_indices_per_block = {
                k: list(v.keys()) for k, v in module.diagonals.items()
            }
            lt_galois = compute_galois_elements_for_linear_transform(
                diag_indices_per_block,
                slots,
                module.bsgs_ratio,
                self.ckks_params.logn,
                self.ckks_params.ring_type,
            )
            galois_elements.update(lt_galois)

        elif isinstance(module, Chebyshev):
            config = {
                "coeffs": list(module.coeffs),
                "basis": "chebyshev",
                "prescale": module.prescale,
                "postscale": getattr(module, "postscale", 1),
                "constant": module.constant,
            }

        elif isinstance(module, Activation):
            config = {
                "coeffs": list(module.coeffs) if module.coeffs is not None else [],
                "basis": "monomial",
                "prescale": 1,
                "postscale": 1,
                "constant": 0,
            }

        elif isinstance(module, Bootstrap):
            elements = module.fhe_input_shape.numel()
            btp_slots = 2 ** math.ceil(math.log2(elements))
            config = {
                "input_level": module.input_level,
                "input_min": float(module.input_min),
                "input_max": float(module.input_max),
                "prescale": module.prescale,
                "postscale": module.postscale,
                "constant": module.constant,
                "slots": btp_slots,
            }
            shape = {}
            if hasattr(module, "fhe_input_shape") and module.fhe_input_shape is not None:
                shape["fhe_input"] = list(module.fhe_input_shape)

        elif isinstance(module, Quad):
            pass  # empty config, depth=1

        elif isinstance(module, (Add, Mult)):
            pass  # empty config

        elif isinstance(module, Flatten):
            pass  # empty config

        return GraphNode(
            name=node_name,
            op=op,
            level=level,
            depth=depth,
            shape=shape if shape else None,
            config=config,
            blob_refs=blob_refs,
        )

    @staticmethod
    def _module_to_op(module):
        """Map module class to op string."""
        if isinstance(module, LinearTransform):
            return "linear_transform"
        if isinstance(module, Quad):
            return "quad"
        if isinstance(module, Chebyshev):
            return "polynomial"
        if isinstance(module, Activation):
            return "polynomial"
        if isinstance(module, Bootstrap):
            return "bootstrap"
        if isinstance(module, Add):
            return "add"
        if isinstance(module, Mult):
            return "mult"
        if isinstance(module, Flatten):
            return "flatten"
        # Unknown module type — skip
        return None

    @staticmethod
    def _extract_edges(network_dag):
        """Extract edges from NetworkDAG, filtering out fork/join nodes.

        For fork nodes: A -> fork -> B, C becomes A -> B and A -> C
        For join nodes: A, B -> join -> C becomes A -> C and B -> C
        """
        edges = []
        # Collect graph-emittable node names: non-fork/join with a known op
        real_nodes = set()
        for node in network_dag.nodes:
            op = network_dag.nodes[node].get("op")
            if op not in ("fork", "join"):
                module = network_dag.nodes[node].get("module")
                if module is not None and Compiler._module_to_op(module) is not None:
                    real_nodes.add(node)

        def _resolve_successors(node):
            """Walk forward through fork/join to find real successors."""
            if node in real_nodes:
                return [node]
            result = []
            for succ in network_dag.successors(node):
                result.extend(_resolve_successors(succ))
            return result

        seen = set()
        for node in real_nodes:
            for succ in network_dag.successors(node):
                for rs in _resolve_successors(succ):
                    edge = (node, rs)
                    if edge not in seen:
                        seen.add(edge)
                        edges.append(GraphEdge(src=node, dst=rs))

        return edges

    def close(self):
        """Release the Go backend. Idempotent."""
        if hasattr(self, "backend") and self.backend:
            self.backend.DeleteScheme()
            self.backend = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
