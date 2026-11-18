# Trusted custom guides

Register a Python factory on a `GuideRegistry` in trusted application code. Its factory must construct a genuine Pyro guide and accept the model plus declared configuration. Supply a typed Pydantic or dataclass schema, version and capabilities where appropriate. Pass the same registry to `BayzFlow` or `from_model_and_data`. Custom code must be installed locally before an MCP client can select it; client arguments only configure JSON values.
