# C3AE Experiments

Two experiments comparing the C3AE age-verification model:

1. **Cleartext quality** — accuracy/FPR/FNR for the ReLU and Quad variants on the UTKFace test split, both overall and within the 16–20 age boundary band.
2. **FHE cost** — forward time and peak RSS for the Quad variant at two CKKS configurations (`logn15`, `logn16`), both at 15 multiplicative levels and no bootstrap. Pure ring-degree comparison.

## Layout

```
experiments/
├── models/      # Python: model defs, train, compile, prep, eval
├── bench/       # Go single-binary FHE bench (subcommands: keygen|encrypt|infer|decrypt)
├── scripts/     # bash orchestrators
├── results/     # generated; only results.md committed
└── build_results.py
```

## How to run

Activate the project venv first. Python entry-points under `models/` are run as modules from this directory (e.g. `python -m models.train ...`) so relative imports resolve.

```sh
cd examples/c3ae-demo/experiments

# Cleartext (Experiment 1) — trains ReLU variant, runs both variants
bash scripts/run_cleartext.sh

# FHE (Experiment 2) — per config
bash scripts/run_fhe.sh logn15
bash scripts/run_fhe.sh logn16

# Build the final report
python build_results.py
```

See `docs/plans/2026-05-08-c3ae-experiments.md` for full design and rationale.
