# EXPCON integration

Bayzflow computes the posterior and writes the metrics, checkpoint and tensor artifacts. EXPCON freezes the inputs, dispatches the workload, verifies its evidence and adjudicates the frozen evaluation. The experiment uses schema 1 and EXPCON 0.2.0. Bayzflow has no runtime dependency on EXPCON.

Prepare a fresh project from a built Bayzflow wheel. The script installs the wheel and dependencies in a CPU venv with uv, records the wheel checksum, and commits the frozen-run inputs. Then run the EXPCON lifecycle from that project.

```bash
bash examples/expcon/prepare.sh dist/bayzflow-0.2.0-py3-none-any.whl /tmp/bayzflow-expcon-demo
cd /tmp/bayzflow-expcon-demo
expcon inspect experiments/gaussian --json
expcon validate experiments/gaussian --json
expcon freeze experiments/gaussian --json
expcon diff experiments/gaussian --json
expcon bundle experiments/gaussian --json
expcon verify-bundle experiments/gaussian/bundle --json
expcon run experiments/gaussian --backend local --json
# Set RUN_ID from the returned run ID.
expcon status "$RUN_ID" --json
expcon fetch "$RUN_ID" --json
expcon artifacts "$RUN_ID" --json
expcon verify "$RUN_ID" --json
expcon adjudicate "$RUN_ID" --json
```

The first frozen attempt could not dispatch because EXPCON did not parse uv's `==3.12.*` constraint. The successor used a compatible range and preserved the model, seed, dataset and acceptance criterion. The final candidate wheel was verified in `v003`, run `run-BAYZFLOW-GAUSSIAN-53f67837`. EXPCON returned `COMPLETED / COMPLIANT / VERIFIED / SUPPORTED` with all required outputs. The worst mean error across the two frozen guides was **0.12218** against the limit **0.35**. The [verification record](evidence/expcon-verification.json) preserves contract, bundle, evidence and artifact identities.

For agent interpretation, supply the verified metrics and adjudication to Pi or Sol-Pi and ask whether they support the frozen posterior mean hypothesis. Keep the claim scoped to this fixture and this threshold. A Pi CLI attempt in an isolated agent directory could not authenticate because no provider API key was available. The EXPCON verification and deterministic adjudication ran independently.
