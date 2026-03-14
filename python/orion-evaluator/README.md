# orion-evaluator

Python bindings to the Orion Go evaluator. Runs FHE inference on compiled models.

## Why keys are passed as bytes

The `Evaluator` constructor accepts serialized `MemEvaluationKeySet` bytes, not Go object handles:

```python
evaluator = Evaluator(params_dict, keys_bytes)  # bytes, not handles
```

This looks weird if you're generating keys and running inference in the same Python process — you serialize a Go object to bytes just to deserialize it back into a Go object on the other side. The reason: Go has shitty FFI.

Each CGO shared library (`.so`) embeds its own Go runtime with its own GC, goroutine scheduler, and `cgo.Handle` table. The `lattigo` package (keygen, encrypt, decrypt) and `orion-evaluator` (inference) are separate `.so` files, so a `cgo.Handle` from one means nothing to the other. They are two isolated Go worlds in the same process. There is no way to share Go pointers across them.

So we serialize to bytes as the common language between the two runtimes. This is also the right interface for real deployments where keys come over the network from a remote client.

## Usage

```python
from orion_evaluator import Model, Evaluator

# Context manager support (recommended — auto-cleanup on block exit)
with Model.load(model_bytes) as model:
    params_dict, manifest, input_level = model.client_params()
    with Evaluator(params_dict, keys_bytes) as evaluator:
        result_ct_bytes = evaluator.forward(model, input_ct_bytes)

# Or explicit close()
model = Model.load(model_bytes)
evaluator = Evaluator(params_dict, keys_bytes)
result_ct_bytes = evaluator.forward(model, input_ct_bytes)
evaluator.close()
model.close()
```

## Errors

All errors raise from `orion_evaluator.errors`:

- `EvaluatorError` — base class for all evaluator errors
- `ModelLoadError(EvaluatorError)` — raised by `Model.load()` on invalid data
