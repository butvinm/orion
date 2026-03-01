"""Compare Python-computed Galois elements against Lattigo-computed ones.

Compiles a SimpleMLP, then for each LinearTransform module:
1. Gets Galois elements from Lattigo (via existing TransformEncoder path)
2. Computes Galois elements in pure Python from diagonal indices
3. Compares the two sets

Also compares power-of-2 Galois elements and output rotation elements.
"""

import sys
import math

import torch

sys.path.insert(0, "/home/butvinm/Dev/orion")

from galois import (
    compute_galois_elements_for_linear_transform,
    galois_element,
    nth_root_for_ring,
)

from orion.params import CKKSParams, CompilerConfig
from orion.compiler import Compiler
from orion.nn.linear import LinearTransform
import orion.nn as on


# ── Model ──────────────────────────────────────────────────────────────

class SimpleMLP(on.Module):
    def __init__(self):
        super().__init__()
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(784, 32)
        self.act1 = on.Quad()
        self.fc2 = on.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        return self.fc2(x)


MLP_PARAMS = CKKSParams(
    logn=13,
    logq=[29, 26, 26, 26, 26, 26],
    logp=[29, 29],
    logscale=26,
    h=8192,
    ring_type="conjugate_invariant",
)


def main():
    torch.manual_seed(42)
    net = SimpleMLP()

    compiler = Compiler(net, MLP_PARAMS)
    compiler.fit(torch.randn(1, 1, 28, 28))

    # ── Build DAG and generate diagonals (same as compile()) ──
    from orion.core.network_dag import NetworkDAG
    from orion.core.fuser import Fuser

    network_dag = NetworkDAG(compiler._traced)
    network_dag.build_dag()

    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
    for module in net.modules():
        if hasattr(module, "update_params") and callable(module.update_params):
            module.update_params()
    for module in net.modules():
        if hasattr(module, "he_mode"):
            module.scheme = compiler

    if compiler.config.fuse_modules:
        fuser = Fuser(network_dag)
        fuser.fuse_modules()
        network_dag.remove_fused_batchnorms()

    topo_sort = list(network_dag.topological_sort())

    last_linear = None
    for node in reversed(topo_sort):
        module = network_dag.nodes[node]["module"]
        if isinstance(module, LinearTransform):
            last_linear = node
            break

    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if isinstance(module, LinearTransform):
            module.generate_diagonals(last=(node == last_linear))

    # Run bootstrap solver to assign levels
    from orion.core.auto_bootstrap import BootstrapSolver, BootstrapPlacer

    network_dag.find_residuals()
    l_eff = len(compiler.params.get_logq()) - 1
    btp_solver = BootstrapSolver(net, network_dag, l_eff=l_eff, context=compiler._context)
    input_level, num_bootstraps, bootstrapper_slots = btp_solver.solve()
    print(f"Bootstrap solver: input_level={input_level}, bootstraps={num_bootstraps}")
    print()

    # ── Compare Galois elements per LinearTransform ──
    slots = compiler.params.get_slots()
    logn = MLP_PARAMS.logn
    ring_type = MLP_PARAMS.ring_type
    nth_root = nth_root_for_ring(logn, ring_type)

    all_lattigo_galois = set()
    all_python_galois = set()

    print(f"Parameters: logn={logn}, slots={slots}, nth_root={nth_root}")
    print(f"Ring type: {MLP_PARAMS.ring_type}")
    print()

    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if not isinstance(module, LinearTransform):
            continue

        print(f"── {node} (bsgs_ratio={module.bsgs_ratio}) ──")

        # Lattigo path: generate transforms, collect Galois elements, delete
        lt_ids = compiler._lt_evaluator.generate_transforms(module)
        lattigo_galois = set()
        for tid in lt_ids.values():
            keys = compiler._lt_evaluator.get_galois_elements(tid)
            lattigo_galois.update(keys)
        compiler._lt_evaluator.delete_transforms(lt_ids)

        # Python path: compute from diagonal indices
        diag_indices_per_block = {}
        for (row, col), diags_dict in module.diagonals.items():
            diag_indices_per_block[(row, col)] = list(diags_dict.keys())

        python_galois = compute_galois_elements_for_linear_transform(
            diag_indices_per_block,
            slots,
            module.bsgs_ratio,
            logn,
            ring_type,
        )

        # Compare
        lattigo_set = set(lattigo_galois)
        python_set = set(python_galois)

        print(f"  Blocks: {len(module.diagonals)}")
        for (row, col), diags_dict in module.diagonals.items():
            print(f"    ({row},{col}): {len(diags_dict)} diagonals")
        print(f"  Lattigo Galois elements: {len(lattigo_set)}")
        print(f"  Python Galois elements:  {len(python_set)}")

        if lattigo_set == python_set:
            print(f"  ✓ MATCH")
        else:
            only_lattigo = lattigo_set - python_set
            only_python = python_set - lattigo_set
            if only_lattigo:
                print(f"  ✗ Only in Lattigo ({len(only_lattigo)}): {sorted(only_lattigo)}")
            if only_python:
                print(f"  ✗ Only in Python ({len(only_python)}): {sorted(only_python)}")

        all_lattigo_galois.update(lattigo_set)
        all_python_galois.update(python_set)
        print()

    # ── Power-of-2 rotations ──
    print("── Power-of-2 rotations ──")
    lattigo_po2 = set()
    i = 1
    while i < slots:
        gal_el = compiler.backend.GetGaloisElement(i)
        lattigo_po2.add(gal_el)
        i *= 2

    python_po2 = set()
    i = 1
    while i < slots:
        python_po2.add(galois_element(i, nth_root))
        i *= 2

    print(f"  Lattigo: {len(lattigo_po2)} elements")
    print(f"  Python:  {len(python_po2)} elements")
    if lattigo_po2 == python_po2:
        print(f"  ✓ MATCH")
    else:
        only_lattigo = lattigo_po2 - python_po2
        only_python = python_po2 - lattigo_po2
        if only_lattigo:
            print(f"  ✗ Only in Lattigo: {sorted(only_lattigo)}")
        if only_python:
            print(f"  ✗ Only in Python: {sorted(only_python)}")
    print()

    # ── Output rotations ──
    print("── Output rotations ──")
    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if isinstance(module, LinearTransform) and module.output_rotations > 0:
            lattigo_out = set()
            python_out = set()
            for i in range(1, module.output_rotations + 1):
                rotation = slots // (2**i)
                lattigo_out.add(compiler.backend.GetGaloisElement(rotation))
                python_out.add(galois_element(rotation, nth_root))

            print(f"  {node}: output_rotations={module.output_rotations}")
            print(f"    Lattigo: {sorted(lattigo_out)}")
            print(f"    Python:  {sorted(python_out)}")
            if lattigo_out == python_out:
                print(f"    ✓ MATCH")
            else:
                print(f"    ✗ MISMATCH")
            print()

    # ── Full manifest comparison ──
    print("=" * 60)
    all_lattigo = all_lattigo_galois | lattigo_po2
    all_python = all_python_galois | python_po2

    # Add output rotation elements
    for node in topo_sort:
        module = network_dag.nodes[node]["module"]
        if isinstance(module, LinearTransform):
            for i in range(1, module.output_rotations + 1):
                rotation = slots // (2**i)
                all_lattigo.add(compiler.backend.GetGaloisElement(rotation))
                all_python.add(galois_element(rotation, nth_root))

    print(f"TOTAL Lattigo Galois elements: {len(all_lattigo)}")
    print(f"TOTAL Python Galois elements:  {len(all_python)}")

    if all_lattigo == all_python:
        print("✓ FULL MATCH — Python implementation produces identical Galois elements")
    else:
        only_lattigo = all_lattigo - all_python
        only_python = all_python - all_lattigo
        print(f"✗ MISMATCH")
        if only_lattigo:
            print(f"  Only in Lattigo ({len(only_lattigo)}): {sorted(only_lattigo)}")
        if only_python:
            print(f"  Only in Python ({len(only_python)}): {sorted(only_python)}")

    compiler.close()


if __name__ == "__main__":
    main()
