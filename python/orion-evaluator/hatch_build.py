import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        plat = os.environ.get("ORION_WHEEL_PLAT")
        if plat:
            build_data["tag"] = f"py3-none-{plat}"
        else:
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
