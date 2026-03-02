"""orion-compiler — Compiles PyTorch neural networks for FHE inference.

Usage:
    from orion_compiler import Compiler, CKKSParams, CompiledModel, KeyManifest
"""

from orion_compiler.params import (
    CKKSParams as CKKSParams,
    CompilerConfig as CompilerConfig,
    CostProfile as CostProfile,
)
from orion_compiler.compiler import Compiler as Compiler
from orion_compiler.compiled_model import (
    CompiledModel as CompiledModel,
    KeyManifest as KeyManifest,
    Graph as Graph,
    GraphNode as GraphNode,
    GraphEdge as GraphEdge,
)

__version__ = "2.0.0"
