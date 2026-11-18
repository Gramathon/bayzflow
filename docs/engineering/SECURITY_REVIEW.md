# Security review

## Controls present in the candidate

- MCP accepts only typed JSON configuration for known guide identifiers. The server does not accept arbitrary Python, dotted import names, local paths, commands, serialized functions, or client-selected checkpoint paths.
- The currently exposed MCP tools are discovery-only. Model operations require objects registered by trusted host Python and do not execute training.
- Posterior weights are loaded using PyTorch `weights_only=True`. Guide parameters are tensor-only; constraints are parsed from JSON against a small fixed set (`real`, zero-bound softplus-positive, greater-than, interval). Unknown constraints fail closed.
- Model state includes Bayesian prior tensors, and metadata records selected layers, prior scales, guide spec and guide digest.
- Bayzflow serializes engine access to Pyro's global parameter store under a process-wide lock and snapshots per-engine state between operations.

## Remaining boundary gaps

The parameter-store lock is not a sandbox and does not coordinate other libraries that use Pyro in the same process. Model code and registered custom guide factories execute as trusted Python. The MCP server does not yet expose bounded jobs, artifacts, or workspace reads, so the issue's filesystem, resource, artifact-identity and job-admission requirements remain unimplemented. There is no external-network transport. No dependency audit, secret scan, static security tool, oversized-file test or adversarial serialization test was executed.

## Assessment

The current discovery interface has a narrow local trust boundary, but the broader production MCP security requirements are not satisfied. Do not expose it as a remote service or describe this candidate as production-ready.
