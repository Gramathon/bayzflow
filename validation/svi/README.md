# SVI engineering validation artifacts

These are direct engineering runs, not EXPCON evidence. Historical E07/E08 records are unchanged.

Run from repository root using the checkout interpreter:

```bash
# Original checkout snapshot, without copying uncommitted files:
mkdir -p /tmp/bayzflow-svi-baseline
git archive 3a559b5343839c0dbb91cbdd1d05a4cb40925b25 bayzflow | tar -x -C /tmp/bayzflow-svi-baseline
.venv/bin/python scripts/svi_gaussian_benchmark.py --source /tmp/bayzflow-svi-baseline --output validation/svi/baseline.json
.venv/bin/python scripts/svi_gaussian_benchmark.py --source /data/dev/repos/VScode/bf-pi-test/.venv/lib/python3.12/site-packages --output validation/svi/installed.json
.venv/bin/python scripts/svi_gaussian_benchmark.py --source "$PWD" --output validation/svi/modified.json
.venv/bin/python scripts/svi_sensitivity.py
.venv/bin/python scripts/svi_record_inputs.py
.venv/bin/python scripts/svi_summarize.py
```

Every main result JSON has source identity, environment, explicit configuration, seed/guide/implementation rows, continuous-checkpoint moments and metrics, all losses, fitting-row predictive summaries and runtime. Sampling count in those analytical guide-moment comparisons is zero for both paths; the separate unit test checks 3000 native draws against guide moments. `summary.json` asserts numerical metric parity and `tables.md` aggregates without changing the underlying results. `checksums.json` covers retained files. Baseline-regression probes can be replayed using `scripts/svi_baseline_regressions.py --source /tmp/bayzflow-svi-baseline`.

Acceptance: direct/Bayzflow checkpoint metrics must agree to absolute 1e-5; exact split continuation must agree bitwise under fixed CPU seeds/batches; sample mean discrepancy divided by sqrt(trace(covariance)) <.08 and sample covariance relative error <.12 for the fixed 3000-draw tests. Those sampling checks establish internal consistency, not exact-target fidelity. Target adequacy is reported independently; none of the default original-feature full-rank fits reaches covariance relative error .10 after 2000 steps.

Known benchmark differences from frozen E08: initialization is explicitly observed before optimization (consumes matched guide draws); the exact reference uses the actual float32-rounded likelihood inputs. E08 used its original double design for reference and initialized inside its first SVI step. The current interpreter uses Torch 2.15.0. Thus the engineering baseline is an E07-style reproduction, not a byte-identical replay of historical E08 trajectories. All arms within a comparison have matched conventions.

`condition_inputs.json` retains design/labels, exact native moments, transformed prior and coordinate matrices with per-array SHA256. Its generator independently checks posterior invariance under every feature transform to 2e-6, accounting for float32 transformation arithmetic. `provenance.json` records final implementation identities; individual result files identify the source present at each run.
