# Floor-Level Localization

Estimate which floor a vehicle is on in a multi-level parking garage using a **Hidden Markov Model (HMM)** and **Viterbi decoding**. A naive “snap to best observation” baseline is included for comparison.

## What’s in the repo

| File | Purpose |
|------|---------|
| `hmm_model.py` | HMM, emissions, Viterbi decode, naive baseline |
| `simulation.py` | Synthetic garage trips, metrics, physical validity checks |
| `main.py` | Hand-worked example + parameter sweep |
| `test_hmm.py` | Unit tests |
| `simulation_results.csv` | Output from the experimental sweep |

## Git hooks (optional)

To block Cursor from adding `Co-authored-by: Cursor` on commits in this repo:

```bash
git config core.hooksPath .githooks
```

The hook is already in `.githooks/commit-msg`.

## Requirements

Python 3. No extra packages—stdlib only.

## Run

```bash
# Hand-worked check + simulation sweep (default: 30 trials per setting)
python main.py

# Fewer trials for a quick run
python main.py 5

# Tests
python -m unittest test_hmm.py
```

The sweep varies floor count, trip length, GPS dropout, and noise level, then compares Viterbi vs. naive snap on accuracy and physically valid floor sequences.

## Results

See `simulation_results.csv` for aggregated metrics. Re-running `main.py` writes a new CSV (currently to `~/Downloads/simulation_results.csv` in `main.py`; copy into the repo if you want it versioned).
