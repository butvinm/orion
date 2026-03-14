from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.fx as fx
import torch.nn as nn

import orion_compiler.nn as on
from orion_compiler.nn.linear import LinearTransform
from orion_compiler.nn.module import Module
from orion_compiler.nn.normalization import BatchNormNd


@dataclass
class NodeStats:
    input_min: float = float("inf")
    input_max: float = float("-inf")
    output_min: float = float("inf")
    output_max: float = float("-inf")
    input_shape: torch.Size | None = None
    output_shape: torch.Size | None = None
    fhe_input_shape: torch.Size | None = None
    fhe_output_shape: torch.Size | None = None
    input_gap: int = 1
    output_gap: int = 1


class OrionTracer(fx.Tracer):
    """
    Overrides the default fx.Tracer that does not recursively access all
    modules in the network. This is a deeper trace.
    """

    def is_leaf_module(self, m: nn.Module, _: str) -> bool:
        if not isinstance(m, nn.Module):
            return False
        if isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        return not any(True for _ in m.children())

    def trace_model(self, model: nn.Module) -> fx.GraphModule:
        # Tracing outputs are slightly different when the user provides
        # a leaf module (e.g on.Conv2d) rather than a network. We'll wrap
        # it temporarily to consistently track FHE statistics.
        if self.is_leaf_module(model, ""):
            model = ModuleWrapper(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fx.GraphModule(model, super().trace(model))


class ModuleWrapper(on.Module):
    """Wrapper for leaf modules to make them traceable."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.module(x)
        return result


class StatsTracker(fx.Interpreter):
    """Tracks important FHE statistics."""

    def __init__(self, module: fx.GraphModule) -> None:
        super().__init__(module)
        self._stats: dict[str, NodeStats] = {}
        self._init_node_attributes()

    @property
    def _graph(self) -> fx.Graph:
        return cast(fx.Graph, self.module.graph)

    def _init_node_attributes(self) -> None:
        # Tracks min/max values and shapes for FHE-friendly inference
        for node in self._graph.nodes:
            self._stats[node.name] = NodeStats()

    def run_node(self, node: fx.Node) -> Any:
        # Run one node and track its input/output stats
        self._validate_node(node)

        inp = self.map_nodes_to_values(node.args, node)
        if inp:
            self.update_input_stats(cast(tuple[Any, ...], inp), node)

        result = super().run_node(node)  # Forward pass the node
        self.update_output_stats(result, node)

        if node.op == "call_module":
            module = self.module.get_submodule(cast(str, node.target))
            if isinstance(module, Module):
                self.sync_module_attributes(node)

        return result

    def _validate_node(self, node: fx.Node) -> None:
        # Validate that the layer works under FHE
        self._validate_shapes_and_gaps(node)

        if node.op == "call_module":
            self._validate_module_properties(node)

    def _validate_shapes_and_gaps(self, node: fx.Node) -> None:
        # Ensure consistent shapes and gaps across inputs
        parents = node.all_input_nodes
        if not parents:
            return

        # Helper function to check consistency
        def check_consistency(attr_name: str, label: str) -> None:
            values = [
                getattr(self._stats[p.name], attr_name)
                for p in parents
                if getattr(self._stats[p.name], attr_name) is not None
            ]
            if len(set(values)) > 1:
                raise ValueError(f"Inconsistent {label} for {node.name}: {set(values)}")

        # Check all required consistencies
        check_consistency("output_shape", "input shapes")
        check_consistency("fhe_output_shape", "FHE shapes")
        check_consistency("output_gap", "input gaps")

    def _validate_module_properties(self, node: fx.Node) -> None:
        # Check module-specific FHE compatibility requirements
        submodule = self.module.get_submodule(cast(str, node.target))

        # Check stride equality in pooling layers
        stride = getattr(submodule, "stride", None)
        if stride and len(set(stride)) > 1:
            raise ValueError(f"Stride for {node.name} must be equal in all directions: {stride}")

        # Check BatchNorm parent count
        is_batchnorm = isinstance(submodule, BatchNormNd)
        has_multiple_parents = len(node.all_input_nodes) > 1

        if is_batchnorm and has_multiple_parents:
            raise ValueError(f"BatchNorm node {node} has multiple parents which prevents fusion")

    def update_input_stats(self, inp: tuple[Any, ...], node: fx.Node) -> None:
        # Update input statistics from actual tensor values
        s = self._stats[node.name]
        min_values = []
        max_values = []

        for e in inp:
            if isinstance(e, torch.Tensor):
                min_values.append(e.detach().min())
                max_values.append(e.detach().max())
            else:  # scalars
                scalar_tensor = torch.tensor(e)
                min_values.append(scalar_tensor)
                max_values.append(scalar_tensor)

        current_min = float(min(min_values))
        current_max = float(max(max_values))
        s.input_min = min(s.input_min, current_min)
        s.input_max = max(s.input_max, current_max)

        # Set input shape from parent's output shape for structure preservation
        if node.all_input_nodes:
            parent = node.all_input_nodes[0]
            ps = self._stats[parent.name]
            s.input_shape = ps.output_shape
            s.input_gap = ps.output_gap
            s.fhe_input_shape = ps.fhe_output_shape
        else:
            # For input nodes with no parents, use actual tensor shape
            s.input_shape = inp[0].shape

    def update_output_stats(self, result: torch.Tensor, node: fx.Node) -> None:
        # Update output statistics based on actual result tensor
        s = self._stats[node.name]
        s.output_min = min(s.output_min, float(result.min()))
        s.output_max = max(s.output_max, float(result.max()))

        # Determine appropriate output shape based on module type
        s.output_shape = self.compute_clear_output_shape(node, result)
        s.fhe_output_shape = self.compute_fhe_output_shape(node)
        s.output_gap = self.compute_fhe_output_gap(node)

    def compute_clear_output_shape(self, node: fx.Node, result: torch.Tensor) -> torch.Size:
        s = self._stats[node.name]
        # Determine output shape, preserving structure except for transforming ops
        if not s.input_shape:
            return result.shape

        # Only LinearTransform modules change the output shape
        if node.op == "call_module":
            module = self.module.get_submodule(cast(str, node.target))
            if isinstance(module, LinearTransform):
                return result.shape

        # For all other modules, preserve the input shape
        return s.input_shape

    def compute_fhe_output_gap(self, node: fx.Node) -> int:
        s = self._stats[node.name]
        if node.op == "call_module":
            module = self.module.get_submodule(cast(str, node.target))
            if isinstance(module, LinearTransform):
                return module.compute_fhe_output_gap(
                    input_gap=s.input_gap,
                    input_shape=s.input_shape,
                    output_shape=s.output_shape,
                )
        return s.input_gap

    def compute_fhe_output_shape(self, node: fx.Node) -> torch.Size | None:
        s = self._stats[node.name]
        if not s.input_shape:
            return s.output_shape

        if node.op == "call_module":
            module = self.module.get_submodule(cast(str, node.target))
            if isinstance(module, LinearTransform):
                shape = module.compute_fhe_output_shape(
                    input_gap=s.input_gap,
                    input_shape=s.input_shape,
                    output_shape=s.output_shape,
                    fhe_input_shape=s.fhe_input_shape,
                    output_gap=s.output_gap,
                    clear_output_shape=s.output_shape,
                )
                return torch.Size(shape)
        return s.fhe_input_shape

    def sync_module_attributes(self, node: fx.Node) -> None:
        # Sync tracked node statistics to the corresponding module
        s = self._stats[node.name]
        module = cast(Module, self.module.get_submodule(cast(str, node.target)))
        module.name = node.name

        # Min/max values
        module.input_min = s.input_min
        module.input_max = s.input_max
        module.output_min = s.output_min
        module.output_max = s.output_max

        # Shapes
        module.input_shape = s.input_shape
        module.output_shape = s.output_shape
        module.fhe_input_shape = s.fhe_input_shape
        module.fhe_output_shape = s.fhe_output_shape

        # Multiplexed gaps
        module.input_gap = s.input_gap
        module.output_gap = s.output_gap

    def update_batch_size(self, batch_size: int) -> None:
        for node in self._graph.nodes:
            if node.op == "call_module":
                module = cast(Module, self.module.get_submodule(cast(str, node.target)))

                shape_attrs = [
                    "input_shape",
                    "output_shape",
                    "fhe_input_shape",
                    "fhe_output_shape",
                ]

                # Update only batch dimension
                for attr in shape_attrs:
                    current_shape = getattr(module, attr)
                    new_shape = torch.Size([batch_size, *list(current_shape[1:])])
                    setattr(module, attr, new_shape)

    def propagate(self, *args: torch.Tensor) -> None:
        # Run the graph with the provided inputs
        self.run(*args)
