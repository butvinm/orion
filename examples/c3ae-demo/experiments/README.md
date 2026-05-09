# C3AE Experiments

Self-contained experiment harness for the C3AE age-verification model. Two
measurements are produced:

1. **Cleartext quality** — accuracy / FPR / FNR for the ReLU and Quad (x²)
   variants of C3AE on the UTKFace test split, both overall and within the
   16–20 age boundary band.
2. **FHE cost** — forward time and peak RSS for the Quad variant under two
   CKKS configurations (`logn15`, `logn16`), both at 15 multiplicative
   levels and **without bootstrap**. Same multiplicative depth in both
   configs isolates the cost of doubling the ring degree from `logn=15`
   to `logn=16`.

All FHE work runs in a single Go binary (`bench`) with subcommands
`keygen | encrypt | infer | decrypt`, so peak RSS measurements aren't
contaminated by Python wrapper overhead.

## Prerequisites

- **Python 3.11+** in the project's `uv` workspace venv at
  `/home/butvinm/Dev/orion/.venv` (run `uv sync` from the repo root).
- **Go 1.22+** with CGO enabled and the Lattigo CGO shared library built
  (`python tools/build_lattigo.py` from the repo root). Required by the
  parent project, not by this `bench` module directly — `bench` is a pure
  Go consumer of `github.com/butvinm/orion/v2/evaluator`.
- **UTKFace dataset** extracted at `./data/UTKFace` (relative to this
  directory), as a flat directory of `<age>_<...>.jpg` files. Used by
  `models.train`, `models.eval`, and `models.prep_input`.

Activate the venv before any Python command:

```sh
source /home/butvinm/Dev/orion/.venv/bin/activate
```

Python entry-points under `models/` are run as modules from this directory
(e.g. `python -m models.train ...`) so that `from models.foo import bar`
resolves against the local package.

## Directory layout

```
experiments/
├── README.md              # this file
├── build_results.py       # aggregates results/ + out/ into results/results.md
├── models/                # Python — model defs + scripts
│   ├── c3ae.py            # true-ReLU torch.nn variant
│   ├── c3ae_fhe.py        # Quad (x²) orion_compiler.nn variant (FHE-compatible)
│   ├── params.py          # CKKSParams for logn15 and logn16
│   ├── train.py           # `python -m models.train --variant {relu,fhe} ...`
│   ├── compile.py         # `python -m models.compile --variant fhe --config <name> ...`
│   ├── prep_input.py      # `python -m models.prep_input --boundary-band` → out/inputs/*.bin
│   └── eval.py            # `python -m models.eval` → results/cleartext.csv
├── bench/                 # Go single-binary FHE bench
│   ├── go.mod             # standalone module, replaces orion/v2 → ../../../..
│   ├── main.go            # subcommand router
│   ├── keygen.go          # `bench keygen --model ... --out <dir>`
│   ├── encrypt.go         # `bench encrypt --model --sk --input --out`
│   ├── infer.go           # `bench infer --model --evk --ct --out --metrics --sample-idx`
│   ├── decrypt.go         # `bench decrypt --model --sk --ct` → stdout JSON
│   └── rss.go             # /proc/self/status → VmHWM
├── scripts/
│   ├── run_cleartext.sh   # train (if needed) + eval → results/cleartext.csv
│   └── run_fhe.sh <cfg>   # compile + keygen + per-sample encrypt/infer/decrypt
├── out/                   # generated; ignored by git
│   ├── weights_<variant>.pth
│   ├── inputs/sample_<idx>.bin    # 12288 float64 little-endian = 98304 B
│   ├── inputs/ground_truth.csv
│   └── <config>/
│       ├── model.orion
│       ├── compile.json           # {compile_s, compile_peak_rss_mb, model_bytes}
│       └── keys/
│           ├── sk.bin
│           ├── evk.bin
│           └── keygen.json        # {keygen_s, evk_bytes}
└── results/               # generated; only results.md committed
    ├── cleartext.csv
    ├── <config>/run.jsonl         # one line per sample: {sample_idx, forward_s, peak_rss_mb, result_ct_bytes}
    ├── <config>/keygen_time.log   # /usr/bin/time -v output
    ├── <config>/infer_time.log
    └── results.md                 # final report (committed)
```

## Build the bench binary

```sh
cd bench && go build && cd ..
```

Produces `bench/bench`. The orchestration scripts rebuild it when stale.

## End-to-end commands

From this directory, with venv activated:

```sh
# Cleartext (Experiment 1) — trains ReLU variant if missing, copies existing FHE weights, runs eval
bash scripts/run_cleartext.sh

# FHE (Experiment 2) — one config per invocation
bash scripts/run_fhe.sh logn15
bash scripts/run_fhe.sh logn16

# Aggregate everything in out/ + results/ into a single markdown report
python build_results.py
```

`run_fhe.sh` is idempotent: re-running skips compile / keygen / per-sample
inference if the corresponding artifacts already exist.

## Caveat: no bootstrap in either config

This experiment runs **without bootstrap** in both `logn15` and `logn16`
configs. An earlier draft considered a `logn=15, Q=415` variant with
bootstrap; it was investigated and dropped because the full bootstrap
circuit's `LogQP` (≈1339 bits = 415 residual + ~558 bootstrap inner Q +
~366 boot_LogP) exceeds the 128-bit-dense security bound of 881 at
`logn=15`, with no documented sparse-key bound at `logn=15` high enough
to cover it.

Both surviving configs are therefore pure no-bootstrap with **identical
multiplicative depth (15 levels)**, so the FHE cost table isolates the
cost of doubling the ring degree from `logn=15` to `logn=16`. See the
"Why no bootstrap?" section of the plan for the full derivation:
`/home/butvinm/Dev/orion/docs/plans/2026-05-08-c3ae-experiments.md`.

## Plan reference

Full design rationale, parameter math, security bounds, data-flow
diagram, and post-completion runbook live in
`/home/butvinm/Dev/orion/docs/plans/2026-05-08-c3ae-experiments.md`.
