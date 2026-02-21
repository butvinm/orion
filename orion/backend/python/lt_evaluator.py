import numpy as np


class TransformEncoder:
    """Compile-time only. Encodes diagonals into LinearTransform objects,
    collects Galois elements. No Go evaluator needed.
    """

    def __init__(self, backend, params):
        self.backend = backend
        self.params = params
        self.required_galois_elements = set()

    def generate_transforms(self, linear_layer):
        diagonals = linear_layer.diagonals
        level = linear_layer.level
        bsgs_ratio = linear_layer.bsgs_ratio

        lintransf_ids = {}
        for (row, col), diags in diagonals.items():
            diags_idxs, diags_data = [], []
            for idx, diag in diags.items():
                diags_idxs.append(idx)
                diags_data.extend(diag)

            lintransf_id = self.backend.GenerateLinearTransform(
                diags_idxs, diags_data, level, bsgs_ratio
            )
            lintransf_ids[(row, col)] = lintransf_id
            self._collect_galois_elements(lintransf_id)

        return lintransf_ids

    def get_galois_elements(self, transform_id):
        return self.backend.GetLinearTransformRotationKeys(transform_id)

    def _collect_galois_elements(self, transform_id):
        keys = self.get_galois_elements(transform_id)
        self.required_galois_elements.update(keys)


class TransformEvaluator:
    """Inference-time only. Evaluates LinearTransform on ciphertexts.
    Needs Go evaluator for rescale.
    """

    def __init__(self, backend, evaluator):
        self.backend = backend
        self.evaluator = evaluator
        self.backend.NewLinearTransformEvaluator()

    def evaluate_transforms(self, linear_layer, in_ctensor):
        from orion.backend.python.tensors import CipherTensor

        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape

        transform_ids = np.array(list(linear_layer.transform_ids.values()))
        cols = len(in_ctensor)
        rows = len(transform_ids) // cols

        transform_ids = transform_ids.reshape(rows, cols)
        cts_out = []
        for i in range(rows):
            ct_out = None
            for j in range(cols):
                t_id = transform_ids[i][j]
                res = self.backend.EvaluateLinearTransform(t_id, in_ctensor.ids[j])
                ct = CipherTensor(in_ctensor.context, res, out_shape, fhe_out_shape)
                ct_out = ct if j == 0 else ct_out + ct

            ct_out_rescaled = self.evaluator.rescale(ct_out.ids[0], in_place=False)
            cts_out.append(ct_out_rescaled)

        return CipherTensor(in_ctensor.context, cts_out, out_shape, fhe_out_shape)

    def delete_transforms(self, transform_ids: dict):
        for tid in transform_ids.values():
            self.backend.DeleteLinearTransform(tid)
