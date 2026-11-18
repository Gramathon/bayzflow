# Security policy

Bayzflow is research software. The current MCP server is local stdio discovery only; it exposes no training, file access, subprocess, or network tools. Register models and custom guide factories in trusted host Python before server creation. MCP clients can select and configure registered guides, but cannot provide executable code or import paths.

Posterior checkpoints must be treated as trusted artifacts. The 0.2.0 format loads tensor-only PyTorch data with `weights_only=True` and parses supported parameter constraints from JSON. Unsupported constraints fail closed. Do not restore old pickle-based checkpoints from untrusted sources.

Pyro uses a process-global parameter store. Bayzflow serializes its own engine operations in a process and restores per-engine parameter snapshots; this is not a security sandbox or a multi-process resource scheduler. Use separately managed worker processes when isolation is required.

Report security issues privately to the repository maintainers rather than filing exploit details in a public issue.
