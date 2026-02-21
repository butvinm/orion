class NewEncryptor:
    def __init__(self, context):
        self.context = context
        self.backend = context.backend
        self.new_encryptor()
        self.new_decryptor()

    def new_encryptor(self):
        self.backend.NewEncryptor()

    def new_decryptor(self):
        self.backend.NewDecryptor()

    def encrypt(self, plaintensor):
        from .tensors import CipherTensor

        ciphertext_ids = []
        for ptxt in plaintensor.ids:
            ciphertext_id = self.backend.Encrypt(ptxt)
            ciphertext_ids.append(ciphertext_id)

        return CipherTensor(
            self.context, ciphertext_ids, plaintensor.shape, plaintensor.on_shape)

    def decrypt(self, ciphertensor):
        from .encoder import PlainTensor

        plaintext_ids = []
        for ctxt in ciphertensor.ids:
            plaintext_id = self.backend.Decrypt(ctxt)
            plaintext_ids.append(plaintext_id)

        return PlainTensor(
           self.context,  plaintext_ids, ciphertensor.shape, ciphertensor.on_shape
        )