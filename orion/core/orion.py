import time
import math
from typing import Union, Dict, Any

import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.backend.lattigo import bindings as lgo
from orion.backend.python import (
    parameters, 
    key_generator,
    encoder, 
    encryptor,
    evaluator, 
    poly_evaluator, 
    lt_evaluator,
    bootstrapper
)

from .tracer import StatsTracker, OrionTracer 
from .fuser import Fuser
from .network_dag import NetworkDAG
from .auto_bootstrap import BootstrapSolver, BootstrapPlacer


class Scheme:
    """
    This Scheme class drives most of the functionality in Orion. It
    configures and manages how our framework interfaces with FHE backends,
    and exposes this functionality to the user through attributes such as
    the encoder, evaluators (linear transform, polynomial, etc.) and
    bootstrappers.

    It also serves two important purposes required before running FHE
    inference: fitting the network and then compiling it. The fit() method
    runs cleartext forward passes through the network to determine per-layer
    input ranges, which are then used to fit polynomial approximations to
    common activation functions (e.g., SiLU, ReLU).

    The compile() function is responsible for all packing of data and
    determines a level management policy by running our automatic bootstrap
    placement algorithm. Once done, each Orion module is automatically
    assigned a level that can then be used in its compilation. This primarily
    includes generating the plaintexts needed for each linear transform.
    """

    def __init__(self):
        self.backend = None
        self.traced = None
        self.keyless = False

    def init_scheme(self, config: Union[str, Dict[str, Any]]):
        """Initializes the scheme."""
        if isinstance(config, str):
            try:
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                raise ValueError(f"Configuration file '{config}' not found.")
        elif not isinstance(config, dict):
            raise TypeError("Config must be a file path (str) or a dictionary.")

        self.params = parameters.NewParameters(config)
        self.backend = self.setup_backend(self.params)

        self.keygen = key_generator.NewKeyGenerator(self)
        self.encoder = encoder.NewEncoder(self)
        self.encryptor = encryptor.NewEncryptor(self)
        self.evaluator = evaluator.NewEvaluator(self)
        self.poly_evaluator = poly_evaluator.NewEvaluator(self)
        self.lt_evaluator = lt_evaluator.NewEvaluator(self)
        self.bootstrapper = bootstrapper.NewEvaluator(self)

        return self

    def init_params_only(self, config: Union[str, Dict[str, Any]]):
        """Initializes the scheme with only params and encoder — no keys.

        Used for keyless compilation where we need to trace the network,
        generate diagonal packing, and collect key requirements without
        generating any cryptographic keys.
        """
        if isinstance(config, str):
            try:
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                raise ValueError(f"Configuration file '{config}' not found.")
        elif not isinstance(config, dict):
            raise TypeError("Config must be a file path (str) or a dictionary.")

        self.params = parameters.NewParameters(config)
        self.backend = self.setup_backend(self.params)
        self.keyless = True

        # Encoder only needs scheme.Params — no keys required.
        self.encoder = encoder.NewEncoder(self)

        # Lt evaluator in keyless mode collects galois elements instead
        # of generating rotation keys.
        self.lt_evaluator = lt_evaluator.NewEvaluator(self, keyless=True)

        return self
    
    def delete_scheme(self):
        if self.backend:
            self.backend.DeleteScheme()
    
    def __del__(self):
        self.delete_scheme()
    
    def __str__(self):
        return str(self.params)
        
    def setup_backend(self, params):
        backend = params.get_backend()
        if backend == "lattigo":
            py_lattigo = lgo.LattigoLibrary()
            py_lattigo.setup_bindings(params)
            return py_lattigo
        elif backend in ("heaan", "openfhe"):
            raise ValueError(f"Backend {backend} not yet supported.")
        else:
            raise ValueError(
                f"Invalid {backend}. Set the backend to Lattigo until "
                f"further notice."
            )

    def encode(self, tensor, level=None, scale=None):
        self._check_initialization()
        return self.encoder.encode(tensor, level, scale)

    def decode(self, ptxt):
        self._check_initialization() 
        return self.encoder.decode(ptxt)

    def encrypt(self, ptxt):
        self._check_initialization() 
        return self.encryptor.encrypt(ptxt)

    def decrypt(self, ctxt):
        self._check_initialization()
        return self.encryptor.decrypt(ctxt)
    
    def fit(self, net, input_data, batch_size=128):
        self._check_initialization()

        net.set_scheme(self)
        net.set_margin(self.params.get_margin())
        
        tracer = OrionTracer()
        traced = tracer.trace_model(net)
        self.traced = traced 

        stats_tracker = StatsTracker(traced)

        #-----------------------------------------#
        #   Populate layers with useful metadata  #
        #-----------------------------------------# 

        # Send input_data to the same device as the model.
        param = next(iter(net.parameters()), None)
        device = param.device if param is not None else torch.device("cpu")

        print("\n{1} Finding per-layer input/output ranges and shapes...", 
              flush=True)
        start = time.time()
        if isinstance(input_data, DataLoader):
            # Users often specify small batch sizes for FHE operations.
            # However, fitting statistics with large datasets would take 
            # unnecessarily long with small batches. To speed this up, we'll 
            # temporarily increase the batch size during the statistics-fitting 
            # step, and then restore the original batch size afterward.
            user_batch_size = input_data.batch_size
            if batch_size > user_batch_size:
                dataset = input_data.dataset
                shuffle = input_data.sampler is None or isinstance(input_data.sampler, RandomSampler)
                
                input_data = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=input_data.num_workers,
                    pin_memory=input_data.pin_memory,
                    drop_last=input_data.drop_last
                )

            # Use this (potentially new) dataloader
            for batch in tqdm(input_data, desc="Processing input data",
                    unit="batch", leave=True):
                stats_tracker.propagate(batch[0].to(device))

            # Now we'll reset the batch size back to what the user specified.
            stats_tracker.update_batch_size(user_batch_size)

        elif isinstance(input_data, torch.Tensor):
            stats_tracker.propagate(input_data.to(device)) 
        else:
            raise ValueError(
                "Input data must be a torch.Tensor or DataLoader, but "
                f"received {type(input_data)}."
            )

        #-------------------------------------#
        #      Fit polynomial activations     #
        #-------------------------------------#

        # Now we can use the statistics we just obtained above to fit
        # all polynomial activation functions.
        print("\n{2} Fitting polynomials... ", end="", flush=True)
        start = time.time()
        for module in net.modules():
            if hasattr(module, "fit") and callable(module.fit):
                module.fit()
        print(f"done! [{time.time()-start:.3f} secs.]")

    def compile(self, net):
        self._check_initialization()

        if self.traced is None:
            raise ValueError(
                "Network has not been fit yet! Before running orion.compile(net) "
                "you must run orion.fit(net, input_data)."
            )

        #------------------------------------------------#
        #   Build DAG representation of neural network   #
        #------------------------------------------------#

        network_dag = NetworkDAG(self.traced)
        network_dag.build_dag()

        # Before fusing, we'll instantiate our own Orion parameters (e.g.
        # weights and biases) that can be fused/modified without affecting
        # the original network's parameters.
        for module in net.modules():
            if (hasattr(module, "init_orion_params") and
                    callable(module.init_orion_params)):
                module.init_orion_params()

        #-------------------------------------#
        #       Resolve pooling kernels       #
        #-------------------------------------#

        # AvgPools are implemented as grouped convolutions in Orion, which
        # are not passed arguments for the number of channels for consistency
        # with PyTorch. We must resolve this after the passes above use
        # torch.nn.functional.
        for module in net.modules():
            if hasattr(module, "update_params") and callable(module.update_params):
                module.update_params()

        #------------------------------------------#
        #   Fuse Orion modules (Conv -> BN, etc)   #
        #------------------------------------------#

        enable_fusing = self.params.get_fuse_modules()
        if enable_fusing:
            fuser = Fuser(network_dag)
            fuser.fuse_modules()
            network_dag.remove_fused_batchnorms()

        #---------------------------------------------#
        #   Pack diagonals of all linear transforms   #
        #---------------------------------------------#

        # Then, we must ensure that there is no junk data left in the slots
        # of the final linear layer (leaking information about partials).
        # This would occur when using the hybrid embedding method. We could
        # use an additional level to zero things out, but instead, we'll
        # just force the last linear layer to use the "square" embedding
        # method which solves this while consuming just one level (albeit
        # usually for more ciphertext rotations).
        topo_sort = list(network_dag.topological_sort())

        last_linear = None
        for node in reversed(topo_sort):
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                last_linear = node
                break

        # Now we can generate the diagonals
        print("\n{3} Generating matrix diagonals...", flush=True)
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                print(f"\nPacking {node}:")
                module.generate_diagonals(last=(node == last_linear))

        #------------------------------#
        #   Find and place bootstraps  #
        #------------------------------#

        network_dag.find_residuals()
        #(save_path="network.png", figsize=(8,30)) # optional plot

        print("\n{4} Running bootstrap placement... ", end="", flush=True)
        start = time.time()
        l_eff = len(self.params.get_logq()) - 1
        btp_solver = BootstrapSolver(net, network_dag, l_eff=l_eff)
        input_level, num_bootstraps, bootstrapper_slots = btp_solver.solve()
        print(f"done! [{time.time()-start:.3f} secs.]", flush=True)
        print(f"├── Network requires {num_bootstraps} bootstrap "
            f"{'operation' if num_bootstraps == 1 else 'operations'}.")

        #btp_solver.plot_shortest_path(
        #    save_path="network-with-levels.png", figsize=(8,30) # optional plot
        #)

        if bootstrapper_slots:
            if self.keyless:
                # In keyless mode, just record the bootstrap slot counts
                # into the manifest instead of generating bootstrap keys.
                slots_str = ", ".join(
                    [str(int(math.log2(slot))) for slot in bootstrapper_slots])
                print(f"├── [keyless] Recorded bootstrap slots for logslots = "
                      f"{slots_str} (skipping key generation)")
            else:
                start = time.time()
                slots_str = ", ".join([str(int(math.log2(slot))) for slot in bootstrapper_slots])
                print(f"├── Generating bootstrappers for logslots = {slots_str} ... ",
                      end="", flush=True)

                # Generate the required (potentially sparse) bootstrappers.
                for slot_count in bootstrapper_slots:
                    self.bootstrapper.generate_bootstrapper(slot_count)
                print(f"done! [{time.time()-start:.3f} secs.]")

        btp_placer = BootstrapPlacer(net, network_dag)
        btp_placer.place_bootstraps()

        #------------------------------------------#
        #   Compile Orion modules in the network   #
        #------------------------------------------#

        print("\n{5} Compiling network layers...", flush=True)
        for node in topo_sort:
            node_attrs = network_dag.nodes[node]
            module = node_attrs["module"]
            if isinstance(module, Module):
                print(f"├── {node} @ level={module.level}", flush=True)
                module.compile()

        # In keyless mode, build and return the key requirements manifest
        if self.keyless:
            manifest = self._build_key_manifest(
                net, topo_sort, network_dag, bootstrapper_slots)
            return input_level, manifest

        return input_level # level at which to encrypt the input.

    def _build_key_manifest(self, net, topo_sort, network_dag, bootstrapper_slots):
        """Collects all key requirements into a manifest dict."""
        galois_elements = set()

        # 1. Linear transform rotation keys (collected by lt_evaluator
        #    in keyless mode)
        galois_elements.update(self.lt_evaluator.required_galois_elements)

        # 2. Power-of-2 rotation keys (from AddPo2RotationKeys in Go)
        #    These are GaloisElement(1), GaloisElement(2), ...,
        #    GaloisElement(MaxSlots/2). We compute them from params.
        max_slots = self.params.get_slots()
        logn = self.params.get_logn()
        ringtype = self.params.get_ringtype()
        # GaloisElement for rotation by k in CKKS standard ring:
        #   galEl = 5^k mod (2*N)
        # For conjugate invariant ring:
        #   galEl = 2*k+1 (simplified)
        # We call the Go backend to compute these.
        po2_elements = self._get_po2_galois_elements()
        galois_elements.update(po2_elements)

        # 3. Hybrid method output rotations
        #    From linear.py:71-72: out.roll(slots // (2**i))
        #    for i in range(1, output_rotations+1)
        slots = self.params.get_slots()
        hybrid_elements = set()
        for node in topo_sort:
            module = network_dag.nodes[node]["module"]
            if isinstance(module, LinearTransform):
                for i in range(1, module.output_rotations + 1):
                    rotation = slots // (2 ** i)
                    galEl = self._rotation_to_galois_element(rotation)
                    hybrid_elements.add(galEl)
        galois_elements.update(hybrid_elements)

        manifest = {
            "galois_elements": sorted(galois_elements),
            "bootstrap_slots": sorted(bootstrapper_slots) if bootstrapper_slots else [],
            "needs_rlk": True,
            "po2_galois_elements": sorted(po2_elements),
            "linear_transform_galois_elements": sorted(
                self.lt_evaluator.required_galois_elements),
            "hybrid_output_galois_elements": sorted(hybrid_elements),
        }
        return manifest

    def _get_po2_galois_elements(self):
        """Compute Galois elements for power-of-2 rotations via Go backend."""
        elements = set()
        max_slots = self.backend.GetMaxSlots()
        i = 1
        while i < max_slots:
            galEl = self.backend.GetGaloisElement(i)
            elements.add(galEl)
            i *= 2
        return elements

    def _rotation_to_galois_element(self, rotation):
        """Convert a rotation amount to a Galois element via Go backend."""
        return self.backend.GetGaloisElement(rotation)

    def _check_initialization(self):
        if self.backend is None:
            raise ValueError(
                "Scheme not initialized. Call `orion.init_scheme()` first.") 
        
scheme = Scheme()
