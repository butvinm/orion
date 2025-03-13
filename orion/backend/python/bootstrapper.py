class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend

    def __del__(self):
        self.backend.DeleteBootstrappers()

    def generate_bootstrapper(self, slots):
        logp = self.scheme.params.get_boot_logp()
        return self.backend.NewBootstrapper(logp, slots)
    
    def bootstrap(self, ctxt, slots):
        return self.backend.Bootstrap(ctxt, slots)