import sys


class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend

    def __del__(self):
        if 'sys' in globals() and sys.modules:
            try:
                self.backend.DeleteBootstrappers()
            except Exception:
                pass

    def generate_bootstrapper(self, slots):
        # We will wait to instantiate any bootstrapper until our bootstrap
        # placement algorithm determines they're necessary.
        logp = self.scheme.params.get_boot_logp()
        return self.backend.NewBootstrapper(logp, slots)
    
    def bootstrap(self, ctxt, slots):
        return self.backend.Bootstrap(ctxt, slots)