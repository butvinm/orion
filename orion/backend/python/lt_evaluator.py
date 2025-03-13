import h5py
import torch

from orion.backend.python.tensors import CipherTensor


class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme 
        self.params = scheme.params
        self.backend = scheme.backend

        self.embed_method = self.params.get_embedding_method()
        self.io_mode = self.params.get_io_mode()
        self.diags_path = self.params.get_diags_path()
        self.keys_path = self.params.get_keys_path()

    def generate_transforms(self, linear_layer):
        diagonals = linear_layer.diagonals 
        level = linear_layer.level
        bsgs_ratio = linear_layer.bsgs_ratio
        name = linear_layer.name

        # Generate all linear transforms block by block.
        lintransf_ids = {}        
        for (row, col), diags in diagonals.items(): 
            diags_idxs, diags_data = [], []
            for idx, diag in diags.items(): 
                diags_idxs.append(idx)
                diags_data.extend(diag)

            lintransf_id = self.backend.GenerateLinearTransform(
                diags_idxs, diags_data, level, bsgs_ratio, row, col, 
                name, self.diags_path, self.keys_path, self.io_mode
            )
            lintransf_ids[(row, col)] = lintransf_id

        return lintransf_ids
    
    def save_transforms(self, linear_layer):
        layer_name = linear_layer.name
        diagonals = linear_layer.diagonals 
        on_bias = linear_layer.on_bias 
        output_rotations = linear_layer.output_rotations 
        input_shape = linear_layer.input_shape 
        output_shape = linear_layer.output_shape
        input_min = linear_layer.input_min
        input_max = linear_layer.input_max
        output_min = linear_layer.output_min 
        output_max = linear_layer.output_max

        print("└── saving... ", end="", flush=True)
        with h5py.File(self.diags_path, "a") as f:
            layer = f.require_group(layer_name)

            layer.create_dataset("embedding_method", data=self.embed_method)
            layer.create_dataset("output_rotations", data=output_rotations)
            layer.create_dataset("on_bias", data=on_bias.numpy())
            layer.create_dataset("input_shape", data=list(input_shape))
            layer.create_dataset("output_shape", data=list(output_shape))
            layer.create_dataset("input_min", data=input_min.item())
            layer.create_dataset("input_max", data=input_max.item())
            layer.create_dataset("output_min", data=output_min.item())
            layer.create_dataset("output_max", data=output_max.item())

            # Set up groups for cleartext diagonals and plaintexts
            diags_group = layer.require_group("diagonals")
            layer.require_group("plaintexts")

            for (row, col), diags in diagonals.items():
                block_idx = f"{row}_{col}"
                block_group = diags_group.create_group(block_idx)
                for diag_idx, diag_data in diags.items():
                    block_group.create_dataset(str(diag_idx), data=diag_data,
                                               compression="gzip", chunks=True)
                    diags[diag_idx] = [] # delete after saving

        print("done!")

    def load_transforms(self, linear_layer):
        self._verify_layer_compatibility(linear_layer)

        layer_name = linear_layer.name
        on_bias = linear_layer.on_bias
        output_rotations = linear_layer.output_rotations

        with h5py.File(self.diags_path, "a") as f:
            layer = f[layer_name]

            # Load the diagonals back into the correct struct
            all_diagonals = {}
            diag_group = layer["diagonals"]
            for block in diag_group:
                row, col = map(int, block.split("_")) # 0_1 -> (0,1)
                diags = {}
                block_group = diag_group[block]
                for diag_idx in block_group:
                    diag_data = block_group[diag_idx][:]
                    diags[int(diag_idx)] = diag_data 
                all_diagonals[(row, col)] = diags

        return all_diagonals, on_bias, output_rotations

    def evaluate_transforms(self, linear_layer, in_ctensor):
        layer_name = linear_layer.name
        out_shape = linear_layer.output_shape
        fhe_out_shape = linear_layer.fhe_output_shape 

        # Order-preserving flatten that can be mapped back to 
        # (row, col) format in backend via len(in_ctensor.ids)
        transform_ids = list(linear_layer.transform_ids.values())
        
        out_ctensor_ids = self.backend.EvaluateLinearTransforms(
            transform_ids, in_ctensor.ids, layer_name,
            self.diags_path, self.keys_path, self.io_mode)

        return CipherTensor(
            self.scheme, out_ctensor_ids, out_shape, fhe_out_shape
        )
            
    def delete_transforms(self, transform_ids: dict):
        for tid in transform_ids.values():
            self.backend.DeleteLinearTransform(tid)

    def _verify_layer_compatibility(self, linear_layer):
        layer_name = linear_layer.name

        # -------- Current network values -------- #

        curr_embed_method = linear_layer.scheme.params.get_embedding_method()
        curr_output_rotations = linear_layer.output_rotations
        curr_on_bias = linear_layer.on_bias
        curr_input_shape = linear_layer.input_shape 
        curr_output_shape = linear_layer.output_shape
        curr_input_min = linear_layer.input_min 
        curr_input_max = linear_layer.input_max
        curr_output_min = linear_layer.output_min
        curr_output_max = linear_layer.output_max

        # ------- Previous network values ------- #

        with h5py.File(self.diags_path, "r") as f:

            # Check if the layer exists in the h5py file
            if layer_name not in f:
                raise ValueError(
                    f"Layer '{layer_name}' not found in file {self.diags_path}. " + 
                    "First set IO mode in parameters YAML file to `save`."
                )
            
            layer = f[layer_name]
            
            last_embed_method = layer["embedding_method"][()].decode("utf-8")
            last_output_rotations = layer["output_rotations"][()]
            last_on_bias = torch.tensor(layer["on_bias"][:])
            last_input_shape = torch.Size(layer["input_shape"][:])
            last_output_shape = torch.Size(layer["output_shape"][:])
            last_input_min = layer["input_min"][()]
            last_input_max = layer["input_max"][()]
            last_output_min = layer["output_min"][()]
            last_output_max = layer["output_max"][()]

            # Check each parameter and collect mismatches
            mismatches = []
                            
            if curr_on_bias.shape != last_on_bias.shape:
                mismatches.append(f"on_bias: shape mismatch")
            elif not torch.allclose(curr_on_bias, last_on_bias):
                mismatches.append(f"on_bias: values mismatch")
            
            # Simple equality checks
            if curr_output_rotations != last_output_rotations:
                mismatches.append(f"output_rotations mismatch")

            if curr_input_shape != last_input_shape:
                mismatches.append(f"input_shape mismatch")
            
            if curr_output_shape != last_output_shape:
                mismatches.append(f"output_shape mismatch")
            
            if curr_embed_method != last_embed_method:
                mismatches.append(f"embedding_method mismatch")
            
            if curr_input_min != last_input_min:
                mismatches.append(f"input_min mismatch")
            
            if curr_input_max != last_input_max:
                mismatches.append(f"input_max mismatch")
            
            if curr_output_min != last_output_min:
                mismatches.append(f"output_min mismatch")
            
            if curr_output_max != last_output_max:
                mismatches.append(f"output_max mismatch")
            
            # If there are mismatches, raise a detailed error
            if mismatches:
                error_msg = "Saved network does not match currently instantiated network: "
                error_msg += ", ".join(mismatches)
                error_msg += ". First set IO mode in parameters YAML file to `save` to "
                error_msg += "override existing data. Then loading will work."
                
                raise ValueError(error_msg)