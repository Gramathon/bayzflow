# Bayzflow MCP on Linux

Install the `mcp` extra. Configure the client to run `bayzflow-mcp --stdio --demo --workspace /operator/approved/private-directory`. The server starts with discovery only if the operator omits the workspace. With `--demo`, it registers the bundled Gaussian regression workload, which uses a fixed dataset and likelihood. The console entry point accepts no client input paths. Connect using the official MCP SDK stdio transport.

To connect Pi 1.5.2, run this from the shell where you installed Bayzflow:

```bash
pi mcp add bayzflow --exposure direct -- "$(command -v bayzflow-mcp)" \
  --stdio --demo --workspace "$HOME/.local/state/bayzflow-demo"
pi mcp list --json
```

The listing should show `state: connected` and the discovered tools.

A trusted Python host can create `BayzflowService(workspace=JobStore(workspace, Limits(...)))`, register a Torch model and a `Workload(version, batch, predict, loss)`, and start `create_server(service).run(transport="stdio")`. Register custom guide factories on the service's guide registry. The model, workload callbacks and custom guide registrations must be importable by a spawned worker. Bump the workload version whenever its data, callbacks or other scientific inputs change.

1. Discover registrations and select exact layer names. Call `bayesian_plan` to check model size, parameter count, selected latent dimension, and guide sizing. This tool performs no training.
2. Submit the unchanged plan request and plan ID to `posterior_fit`. The ID binds a model state digest, guide configuration, selection and seed. An idempotency key can replay the same request. An attempt to reuse the key with different inputs is rejected.
3. Poll `job_status`, or request cancellation with `job_cancel`. Inspect a successful checkpoint via `posterior_inspect` and pass its fit job ID to `posterior_sample`, `measure_epistemic`, or `measure_layers`.
4. Download artifacts by job and artifact ID. Concatenate base64 decoded pages using `next_offset` until `eof`, then check the returned SHA256 before using the output. Each response contains at most 32 KiB of decoded content.

The defaults admit one concurrent CPU job, 100 stored jobs, 250 epochs, 256 samples, 100,000 parameters, 8 MiB of model data, 8 MiB of published artifacts, 8 GiB of worker virtual address space, and 45 seconds elapsed time. A workspace is an operator-owned directory with owner-only permissions. On restart, job state and artifact indices persist. An orphaned active job expires at its deadline and must be submitted again with a new idempotency key. Worker limits restrict a single job under Linux and do not make trusted adapter code a sandbox. Keep the workspace and registrations under operator control.

The predictive measurement reports variance of predictions as weights vary and excludes observation noise. Layer measurement reports latent parameter variance. For activation summaries use the Python layer recorder and label the result as activation variability. Neither attention entropy nor prior calibration is evidence of calibrated prediction uncertainty.
