class NewKeyGenerator:
    def __init__(self, scheme):
        self.backend = scheme.backend
        self.new_key_generator()

    def new_key_generator(self):
        self.backend.NewKeyGenerator()
        self.backend.GenerateSecretKey()
        self.backend.GeneratePublicKey()
        self.backend.GenerateRelinearizationKey()
        self.backend.GenerateEvaluationKeys()
