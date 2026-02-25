"""Evaluator: loads compiled model, keys, and runs FHE inference.

No secret key. Reconstructs module state from CompiledModel metadata,
loads evaluation keys, and runs the forward pass on Ciphertexts.
Uses the orionclient FFI bridge (handle-based, no global state).
"""

import types

import numpy as np
import torch

from orion.compiled_model import CompiledModel, EvalKeys
from orion.ciphertext import Ciphertext, PlainText
from orion.nn.module import Module
from orion.nn.linear import LinearTransform
from orion.nn.operations import Bootstrap
from orion.backend.orionclient import ffi


class _EvalContext:
    """Lightweight context object passed through CipherTensors/Ciphertexts.

    Replaces the SimpleNamespace with 5 evaluator objects. Now just
    carries the evaluator FFI handle plus param info needed by nn modules.
    """

    def __init__(self, eval_handle, ckks_params, compiled):
        self.eval_handle = eval_handle
        self.ckks_params = ckks_params
        self.margin = compiled.config.margin
        self.config = compiled.config
        # Get actual moduli and scale from Go (NTT-friendly primes, not 2^logq)
        self._moduli_chain = ffi.eval_moduli_chain(eval_handle)
        self._default_scale = ffi.eval_default_scale(eval_handle)
        self._max_slots = ckks_params.max_slots

        # Compatibility aliases: nn modules access context.encoder,
        # context.poly_evaluator, context.lt_evaluator, context.params.
        # This object serves all those roles.
        self.encoder = self
        self.poly_evaluator = self
        self.lt_evaluator = self
        self.params = self
        self.evaluator = self
        self.bootstrapper = self
        self.encryptor = None  # No secret key on server
        self.backend = None  # Not used in new FFI path

    def encode(self, values, level, scale=None):
        """Encode values into a PlainText using the evaluator's encoder."""
        if isinstance(values, torch.Tensor):
            values = values.cpu()

        if scale is None:
            scale = self._default_scale

        max_slots = self._max_slots
        num_elements = values.numel() if isinstance(values, torch.Tensor) else len(values)

        if isinstance(values, torch.Tensor):
            pad_length = (-num_elements) % max_slots
            vector = torch.zeros(num_elements + pad_length, dtype=torch.float64)
            vector[:num_elements] = values.flatten().to(torch.float64)
        else:
            pad_length = (-num_elements) % max_slots
            vector = [0.0] * (num_elements + pad_length)
            for i, v in enumerate(values):
                vector[i] = float(v)

        to_encode = vector.tolist() if isinstance(vector, torch.Tensor) else vector
        pt_h = ffi.eval_encode(self.eval_handle, to_encode, level, scale)
        shape = values.shape if isinstance(values, torch.Tensor) else [len(values)]
        return PlainText(pt_h, shape=shape)

    def get_moduli_chain(self):
        """Get the actual moduli chain (NTT-friendly primes from Go)."""
        return self._moduli_chain

    def get_default_scale(self):
        return self._default_scale

    def get_slots(self):
        return self._max_slots

    def get_debug_status(self):
        return False

    def generate_monomial(self, coeffs):
        """Generate a monomial polynomial handle."""
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return ffi.generate_polynomial_monomial(coeffs[::-1])

    def generate_chebyshev(self, coeffs):
        """Generate a Chebyshev polynomial handle."""
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return ffi.generate_polynomial_chebyshev(coeffs)

    def evaluate_polynomial(self, ct, poly_h, out_scale=None):
        """Evaluate a polynomial on a ciphertext."""
        if out_scale is None:
            out_scale = self._default_scale
        r = ffi.eval_poly(self.eval_handle, ct._handle, poly_h, int(out_scale))
        result = Ciphertext(r, shape=ct.shape, context=self)
        result.on_shape = ct.on_shape
        return result

    def evaluate_transforms(self, linear_layer, in_ct):
        """Evaluate linear transforms on a ciphertext.

        Replaces TransformEvaluator.evaluate_transforms.
        """
        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape

        transform_handles = linear_layer.transform_handles
        keys = list(transform_handles.keys())
        cols = max(k[1] for k in keys) + 1 if keys else 1
        rows = max(k[0] for k in keys) + 1 if keys else 1

        # For single-ct input, evaluate each row's transforms
        cts_out_handles = []
        ct_out_h = None
        try:
            for i in range(rows):
                ct_out_h = None
                for j in range(cols):
                    lt_h = transform_handles.get((i, j))
                    if lt_h is None:
                        continue
                    res_h = ffi.eval_linear_transform(
                        self.eval_handle, in_ct._handle, lt_h
                    )
                    if ct_out_h is None:
                        ct_out_h = res_h
                    else:
                        # Add results
                        combined = ffi.eval_add(self.eval_handle, ct_out_h, res_h)
                        ct_out_h.close()
                        res_h.close()
                        ct_out_h = combined

                if ct_out_h is None:
                    continue

                # Rescale
                rescaled_h = ffi.eval_rescale(self.eval_handle, ct_out_h)
                ct_out_h.close()
                ct_out_h = None
                cts_out_handles.append(rescaled_h)
        except:
            # Clean up intermediate and accumulated handles on error
            if ct_out_h is not None:
                ct_out_h.close()
            for h in cts_out_handles:
                h.close()
            raise

        # For now, assuming single output CT
        if len(cts_out_handles) == 1:
            result = Ciphertext(
                cts_out_handles[0], shape=out_shape, context=self,
            )
            result.on_shape = fhe_out_shape
            return result

        # Multiple rows: would need multi-ct support.
        # For current use, return first (nn modules handle single-ct).
        # Free unused handles to avoid leaking cgo handles.
        for h in cts_out_handles[1:]:
            h.close()
        result = Ciphertext(
            cts_out_handles[0], shape=out_shape, context=self,
        )
        result.on_shape = fhe_out_shape
        return result



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
        self._eval_handle = None
        self._tracked_handles = []

        try:
            pj = compiled.params.to_bridge_json()

            # Build the EvalKeyBundle handle from serialized keys
            keys_handle = ffi.new_eval_key_bundle()
            try:
                if keys.has_rlk:
                    ffi.eval_key_bundle_set_rlk(keys_handle, keys.rlk_data)

                for gal_el, key_data in keys.galois_keys.items():
                    ffi.eval_key_bundle_add_galois_key(keys_handle, gal_el, key_data)

                for slot_count, key_data in keys.bootstrap_keys.items():
                    ffi.eval_key_bundle_add_bootstrap_key(keys_handle, slot_count, key_data)

                if compiled.manifest.boot_logp:
                    ffi.eval_key_bundle_set_boot_logp(
                        keys_handle, list(compiled.manifest.boot_logp)
                    )

                # Create the Go Evaluator (loads all keys internally)
                self._eval_handle = ffi.new_evaluator(pj, keys_handle)
            finally:
                # Bundle data copied into Go Evaluator (or error); always free
                keys_handle.close()

            # Build inference context
            self._context = _EvalContext(self._eval_handle, compiled.params, compiled)

            # Reconstruct module state from CompiledModel
            self._reconstruct_modules(compiled)
        except:
            self.close()
            raise

    def _reconstruct_modules(self, compiled: CompiledModel):
        """Apply CompiledModel metadata to the network skeleton."""
        net = self.net
        meta = compiled.module_metadata
        blobs = compiled.blobs

        module_map = dict(net.named_modules())

        for mod_name, mod_meta in meta.items():
            if mod_meta["type"] == "Bootstrap" and "hook_target" in mod_meta:
                continue

            if mod_name not in module_map:
                continue

            module = module_map[mod_name]

            if "level" in mod_meta:
                module.set_level(mod_meta["level"])
            if "depth" in mod_meta:
                module.set_depth(mod_meta["depth"])
            if "fused" in mod_meta:
                module.fused = mod_meta["fused"]

            if mod_meta.get("fhe_input_shape") is not None:
                module.fhe_input_shape = torch.Size(mod_meta["fhe_input_shape"])
            if mod_meta.get("fhe_output_shape") is not None:
                module.fhe_output_shape = torch.Size(mod_meta["fhe_output_shape"])
            if mod_meta.get("output_shape") is not None:
                module.output_shape = torch.Size(mod_meta["output_shape"])
            if mod_meta.get("input_shape") is not None:
                module.input_shape = torch.Size(mod_meta["input_shape"])

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

        self._reconstruct_bootstrap_hooks(net, meta, module_map)

    def _reconstruct_linear(self, module, meta, blobs):
        """Reconstruct a LinearTransform module from metadata + blobs."""
        transform_handles = {}
        for key_str, blob_idx in meta["transform_blobs"].items():
            row, col = map(int, key_str.split(","))
            blob_data = blobs[blob_idx]
            lt_h = ffi.linear_transform_unmarshal(blob_data)
            transform_handles[(row, col)] = lt_h
            self._tracked_handles.append(lt_h)

        module.transform_handles = transform_handles
        module.bsgs_ratio = meta["bsgs_ratio"]
        module.output_rotations = meta["output_rotations"]

        # Re-encode bias using the evaluator's encoder
        bias_blob = blobs[meta["bias_blob"]]
        bias_vec = np.frombuffer(bias_blob, dtype=np.float64)
        bias_tensor = torch.tensor(bias_vec, dtype=torch.float64)
        module.on_bias_ptxt = self._context.encode(
            bias_tensor, level=module.level - module.depth
        )
        self._tracked_handles.append(module.on_bias_ptxt._handle)
        module._eval_context = self._context

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
        module.compile(self._context)
        self._tracked_handles.append(module.poly)

    def _reconstruct_activation(self, module, meta):
        """Reconstruct a monomial Activation from metadata."""
        if meta.get("coeffs") is not None:
            module.coeffs = meta["coeffs"]
        module.output_scale = meta.get("output_scale")
        module.compile(self._context)
        self._tracked_handles.append(module.poly)

    def _reconstruct_relu(self, module, meta):
        module.prescale = meta["prescale"]
        module.postscale = meta["postscale"]

    def _reconstruct_batchnorm(self, module, meta):
        if meta.get("fused", False):
            return
        if hasattr(module, "init_orion_params"):
            module.init_orion_params()
        module.compile(self._context)
        # Track all PlainText handles created during BatchNorm compile
        for attr in ('on_running_mean_ptxt', 'on_inv_running_std_ptxt',
                     'on_weight_ptxt', 'on_bias_ptxt'):
            pt = getattr(module, attr, None)
            if pt is not None and hasattr(pt, '_handle'):
                self._tracked_handles.append(pt._handle)

    def _reconstruct_bootstrap_hooks(self, net, meta, module_map):
        """Recreate bootstrap forward hooks from metadata."""
        for mod_name, mod_meta in meta.items():
            if mod_meta["type"] != "Bootstrap" or "hook_target" not in mod_meta:
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

            bootstrapper.compile(self._context)
            self._tracked_handles.append(bootstrapper.prescale_ptxt._handle)

            target_module.bootstrapper = bootstrapper
            target_module.register_forward_hook(
                lambda mod, inp, out, btp=bootstrapper: btp(out)
            )

    def run(self, ct: Ciphertext) -> Ciphertext:
        """Run FHE inference on an encrypted input.

        Sets the evaluator context on the ciphertext and runs the forward pass.
        """
        ct.context = self._context
        ct.on_shape = ct.shape

        self.net.he()
        out = self.net(ct)

        # The output is already a Ciphertext with context set
        return out

    def close(self):
        # Close all tracked reconstruction handles (LT, poly, bias)
        for h in getattr(self, '_tracked_handles', []):
            try:
                h.close()
            except Exception:
                pass
        self._tracked_handles = []

        # Close the Go Evaluator (two-step: resource cleanup + handle delete)
        if hasattr(self, '_eval_handle') and self._eval_handle:
            ffi.evaluator_close(self._eval_handle)
            self._eval_handle.close()
            self._eval_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, '_eval_handle') and self._eval_handle:
            try:
                self.close()
            except Exception:
                pass
