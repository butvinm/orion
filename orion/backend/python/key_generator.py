import h5py 

class NewKeyGenerator:
    def __init__(self, scheme):
        self.backend = scheme.backend
        self.io_mode = scheme.params.get_io_mode()
        self.keys_path = scheme.params.get_keys_path()
        self.new_key_generator()

    def new_key_generator(self):
        self.backend.NewKeyGenerator()
        self.generate_secret_key()
        self.generate_public_key()
        self.generate_relinearization_key()
        self.generate_evaluation_keys()

    def generate_secret_key(self):
        if self.io_mode != "load": # we'll need to generate a fresh sk
            self.backend.GenerateSecretKey()
            
            # Save key if in "save" mode
            if self.io_mode == "save":
                sk_serial, _ = self.backend.SerializeSecretKey()
                with h5py.File(self.keys_path, "a") as f:
                    f.create_dataset("sk", data=sk_serial)
        
        # Load key if in "load" mode
        elif self.io_mode == "load":
            with h5py.File(self.keys_path, "r") as f:
                serial_sk = f["sk"][()]
                self.backend.LoadSecretKey(serial_sk)

    def generate_public_key(self):
        self.backend.GeneratePublicKey()

    def generate_relinearization_key(self):
        self.backend.GenerateRelinearizationKey()

    def generate_evaluation_keys(self):
        self.backend.GenerateEvaluationKeys()