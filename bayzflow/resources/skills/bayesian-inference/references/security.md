# Security

Only operate on trusted in-process models and registered guide factories. The stdio MCP server has no filesystem or subprocess tools. Do not add arbitrary path, import, pickle, or code execution arguments. Pyro's parameter store is process global, so Bayzflow operations are serialized within a process; use separate worker processes for parallel jobs and manage their resources externally.
