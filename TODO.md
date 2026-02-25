- [x] Split client, compiler and evaluator, extract keys generation from main workflow and allow to accept keys from lattigo
- [x] Generate keys one-by-one
- [ ] Get rid of python client, provide only Go API
- [x] Create WASM FHE demo (browser-based encrypted inference)
- [ ] Add threshold encryption to WASM demo
- [ ] Store computation graph in compiled model, implement pure go evaluator

My vague view on final architecture:

1. lattigo (Golang)
   - thin python and js bindings (for python and wasm demos and use in my future project)
   - those bindings are not related to Orion directly, anyone can implement lattigo bindings for any language, but we store them in repo for our demos
2. Orion Client (Golang)
   - encoding/decoding of tensors into plaintexts, essentially one list of number to another, nothing else
   - thin python and js bindings (for demos, but again, who would use orion can write bindings to any language)
3. Orion Compiler and Torch modules
   - Torch modules for Orion-specific blocks like on.Conv2d
   - Python compiler that fits data into model to find crypto params, traverse model to build computation graph and packed diagonals and other model params, can serialize model, inference and crypto params
4. Orion Evaluator (Golang)
   - Get params and model, performs inference
   - Accepts lattigo crypto context FROM USER! This is essential to be independent from scheme, so user can enable threshold encryption or whatever he wants, we only responsible to perform operations on ciphertexts in proper order
