"""Evaluator: loads compiled model, keys, and runs FHE inference.

No secret key. Reconstructs module state from CompiledModel metadata,
loads evaluation keys, and runs the forward pass on CipherTexts.
"""

import types

import numpy as np
import torch

from orion.compiled_model import CompiledModel, EvalKeys
from orion.client import CipherText
from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.nn.operations import Bootstrap
from orion.backend.lattigo import bindings as lgo
from orion.backend.python import parameters, encoder
from orion.backend.python.evaluator import NewEvaluator as EvalWrapper
from orion.backend.python.poly_evaluator import PolynomialEvaluator
from orion.backend.python.lt_evaluator import TransformEvaluator
from orion.backend.python.bootstrapper import NewEvaluator as BootstrapperEvaluator
from orion.backend.python.tensors import CipherTensor


class Evaluator:
    """Server-side FHE inference engine.

    Usage:
        evaluator = Evaluator(net, compiled, keys)
        ct_result = evaluator.run(ct)
    """

    def __init__(
        self,
        net: Module,
        compiled: CompiledModel,
        keys: EvalKeys,
    ):
        self.net = net
        self.compiled = compiled
        self.ckks_params = compiled.params

        # 1. Init Go backend with params (no keys yet)
        self.params = parameters.NewParameters.from_ckks_params(
            compiled.params, compiled.config
        )
        self.backend = lgo.LattigoLibrary()
        self.backend.setup_bindings(self.params)

        # 2. Create encoder
        self._encoder = encoder.NewEncoder(self)

        # 3. Load keys into Go backend
        self._load_keys(keys)

        # 4. Create evaluator from loaded keys
        self.backend.NewEvaluatorFromKeys()

        # 5. Create inference-time wrappers (MUST happen after step 4)
        # Create Python evaluator wrapper WITHOUT calling Go NewEvaluator()
        # (which would overwrite our loaded keys). The Go evaluator was
        # already created by NewEvaluatorFromKeys() above.
        self._evaluator = EvalWrapper.__new__(EvalWrapper)
        self._evaluator.backend = self.backend
        self._lt_evaluator = TransformEvaluator(
            self.backend, self._evaluator
        )
        self._poly_evaluator = PolynomialEvaluator(self.backend, self.params)

        # 6. Create bootstrapper evaluators
        self._bootstrapper = BootstrapperEvaluator(self)
        self._init_bootstrappers(compiled)

        # 7. Build inference context
        self._context = self._build_context()

        # 8. Reconstruct module state from CompiledModel
        self._reconstruct_modules(compiled)

    # -- Properties expected by wrappers that take a "scheme" --

    @property
    def encoder(self):
        return self._encoder

    @property
    def encryptor(self):
        return None  # Server has no secret key

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def lt_evaluator(self):
        return self._lt_evaluator

    @property
    def poly_evaluator(self):
        return self._poly_evaluator

    @property
    def bootstrapper(self):
        return self._bootstrapper

    def _load_keys(self, keys: EvalKeys):
        """Load all evaluation keys into the Go backend.

        Order matters: LoadRelinKey -> GenerateEvaluationKeys (creates
        the EvalKeys struct) -> LoadRotationKey (populates GaloisKeys map).
        """
        if keys.has_rlk:
            rlk_arr = np.frombuffer(keys.rlk_data, dtype=np.uint8)
            self.backend.LoadRelinKey(rlk_arr)

        # Create EvalKeys struct from loaded RelinKey. Must happen
        # BEFORE loading rotation keys, which populate EvalKeys.GaloisKeys.
        self.backend.GenerateEvaluationKeys()

        for gal_el, key_data in keys.galois_keys.items():
            key_arr = np.frombuffer(key_data, dtype=np.uint8)
            self.backend.LoadRotationKey(key_arr, gal_el)

        for slot_count, key_data in keys.bootstrap_keys.items():
            key_arr = np.frombuffer(key_data, dtype=np.uint8)
            logp = list(self.compiled.manifest.boot_logp)
            self.backend.LoadBootstrapKeys(key_arr, slot_count, logp)

    def _init_bootstrappers(self, compiled: CompiledModel):
        """Initialize bootstrapper evaluators for required slot counts."""
        if compiled.manifest.bootstrap_slots:
            for slot_count in compiled.manifest.bootstrap_slots:
                self._bootstrapper.generate_bootstrapper(slot_count)

    def _build_context(self):
        """Build the inference context namespace for CipherTensors."""
        ctx = types.SimpleNamespace()
        ctx.backend = self.backend
        ctx.params = self.params
        ctx.encoder = self._encoder
        ctx.encryptor = None  # No secret key on server
        ctx.evaluator = self._evaluator
        ctx.lt_evaluator = self._lt_evaluator
        ctx.poly_evaluator = self._poly_evaluator
        ctx.bootstrapper = self._bootstrapper
        ctx.margin = self.compiled.config.margin
        ctx.config = self.compiled.config
        return ctx

    def _reconstruct_modules(self, compiled: CompiledModel):
        """Apply CompiledModel metadata to the network skeleton."""
        net = self.net
        meta = compiled.module_metadata
        blobs = compiled.blobs

        # Walk all named modules and apply metadata
        module_map = dict(net.named_modules())

        for mod_name, mod_meta in meta.items():
            # Skip bootstrap hooks -- handled separately below
            if mod_meta["type"] == "Bootstrap" and "hook_target" in mod_meta:
                continue

            if mod_name not in module_map:
                continue

            module = module_map[mod_name]

            # Set level/depth
            if "level" in mod_meta:
                module.set_level(mod_meta["level"])
            if "depth" in mod_meta:
                module.set_depth(mod_meta["depth"])
            if "fused" in mod_meta:
                module.fused = mod_meta["fused"]

            # Set FHE shapes
            if mod_meta.get("fhe_input_shape") is not None:
                module.fhe_input_shape = torch.Size(
                    mod_meta["fhe_input_shape"]
                )
            if mod_meta.get("fhe_output_shape") is not None:
                module.fhe_output_shape = torch.Size(
                    mod_meta["fhe_output_shape"]
                )
            if mod_meta.get("output_shape") is not None:
                module.output_shape = torch.Size(mod_meta["output_shape"])
            if mod_meta.get("input_shape") is not None:
                module.input_shape = torch.Size(mod_meta["input_shape"])

            # Type-specific reconstruction
            if isinstance(module, LinearTransform):
                self._reconstruct_linear(module, mod_meta, blobs)

            elif mod_meta["type"] == "Chebyshev":
                self._reconstruct_chebyshev(module, mod_meta)

            elif mod_meta["type"] == "Activation":
                self._reconstruct_activation(module, mod_meta)

            elif mod_meta["type"] in ("BatchNorm1d", "BatchNorm2d"):
                self._reconstruct_batchnorm(module, mod_meta)

            elif mod_meta["type"] == "ReLU":
                self._reconstruct_relu(module, mod_meta)

        # Reconstruct bootstrap hooks
        self._reconstruct_bootstrap_hooks(net, meta, module_map)

    def _reconstruct_linear(self, module, meta, blobs):
        """Reconstruct a LinearTransform module from metadata + blobs."""
        # Load serialized LinearTransform objects
        transform_ids = {}
        for key_str, blob_idx in meta["transform_blobs"].items():
            row, col = map(int, key_str.split(","))
            blob_data = blobs[blob_idx]
            blob_arr = np.frombuffer(blob_data, dtype=np.uint8)
            tid = self.backend.LoadLinearTransform(blob_arr)
            transform_ids[(row, col)] = tid

        module.transform_ids = transform_ids
        module._lt_evaluator = self._lt_evaluator
        module.bsgs_ratio = meta["bsgs_ratio"]
        module.output_rotations = meta["output_rotations"]

        # Decode and re-encode bias
        bias_blob = blobs[meta["bias_blob"]]
        bias_vec = np.frombuffer(bias_blob, dtype=np.float64)
        bias_tensor = torch.tensor(bias_vec, dtype=torch.float64)
        module.on_bias_ptxt = self._encoder.encode(
            bias_tensor, level=module.level - module.depth
        )

    def _reconstruct_chebyshev(self, module, meta):
        """Reconstruct a Chebyshev activation from metadata."""
        module.coeffs = meta["coeffs"]
        module.prescale = meta["prescale"]
        module.constant = meta["constant"]
        module.output_scale = meta.get("output_scale")
        module.degree = meta["degree"]
        if meta.get("input_min") is not None:
            module.input_min = meta["input_min"]
        if meta.get("input_max") is not None:
            module.input_max = meta["input_max"]

        # Regenerate polynomial object
        module.compile(self._context)

    def _reconstruct_activation(self, module, meta):
        """Reconstruct a monomial Activation from metadata."""
        if meta.get("coeffs") is not None:
            module.coeffs = meta["coeffs"]
        module.output_scale = meta.get("output_scale")

        # Regenerate polynomial object
        module.compile(self._context)

    def _reconstruct_relu(self, module, meta):
        """Reconstruct a ReLU module from metadata."""
        module.prescale = meta["prescale"]
        module.postscale = meta["postscale"]

    def _reconstruct_batchnorm(self, module, meta):
        """Reconstruct a BatchNorm module."""
        if meta.get("fused", False):
            # Fused BN is skipped during forward -- no need to encode plaintexts
            return
        # Non-fused BatchNorm needs init_orion_params + compile
        if hasattr(module, "init_orion_params"):
            module.init_orion_params()
        module.compile(self._context)

    def _reconstruct_bootstrap_hooks(self, net, meta, module_map):
        """Recreate bootstrap forward hooks from metadata."""
        for mod_name, mod_meta in meta.items():
            if (
                mod_meta["type"] != "Bootstrap"
                or "hook_target" not in mod_meta
            ):
                continue

            target_name = mod_meta["hook_target"]
            if target_name not in module_map:
                continue

            target_module = module_map[target_name]

            bootstrapper = Bootstrap(
                input_min=mod_meta["input_min"],
                input_max=mod_meta["input_max"],
                input_level=mod_meta["input_level"],
            )
            bootstrapper.prescale = mod_meta["prescale"]
            bootstrapper.postscale = mod_meta["postscale"]
            bootstrapper.constant = mod_meta["constant"]

            if mod_meta.get("fhe_input_shape") is not None:
                bootstrapper.fhe_input_shape = torch.Size(
                    mod_meta["fhe_input_shape"]
                )

            # Compile the bootstrapper (creates prescale_ptxt)
            bootstrapper.compile(self._context)

            target_module.bootstrapper = bootstrapper
            target_module.register_forward_hook(
                lambda mod, inp, out, btp=bootstrapper: btp(out)
            )

    def run(self, ct: CipherText) -> CipherText:
        """Run FHE inference on an encrypted input.

        Converts CipherText to CipherTensor, runs the forward pass,
        and converts the output back to CipherText.
        """
        # Convert CipherText (client wrapper) to CipherTensor (backend tensor)
        # Clear CipherText ids so its eventual GC doesn't delete Go ciphertexts
        in_ids = ct.ids
        ct.ids = []
        in_ctensor = CipherTensor(
            self._context, in_ids, ct.shape
        )

        # Switch to HE mode and run
        self.net.he()
        out_ctensor = self.net(in_ctensor)

        # Take ownership of output IDs from CipherTensor to prevent
        # CipherTensor.__del__ from deleting them on the Go heap
        out_ids = list(out_ctensor.ids)
        out_shape = out_ctensor.shape
        out_ctensor.ids = []

        return CipherText(out_ids, out_shape, self.backend)

    def __del__(self):
        if hasattr(self, "backend") and self.backend:
            try:
                self.backend.DeleteScheme()
            except Exception:
                pass
