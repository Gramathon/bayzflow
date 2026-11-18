# Packaging validation

`uv build` built the 0.2.0 wheel and sdist. `twine check --strict` passed for both. The candidate was installed into a fresh Python 3.12.13 CPU venv outside the repository with PyTorch 2.15.0+cpu, Pyro 1.9.2 and MCP 2.3.0. Package imports resolved to the venv's site-packages. Version, bundled model inspection and skill install passed. The full checked-in pytest suite against the installed wheel returned **22 passed**. This included an installed MCP child process, real CPU inference and checkpoint reload without the checkout on its import path.

| Local candidate file | SHA256 |
|---|---|
| `bayzflow-0.2.0-py3-none-any.whl` | `7a6686dc0c687f29958de90cf932bf14d7f817d9790133056d3b1a43409393b2` |
| `bayzflow-0.2.0.tar.gz` | `7b8dab90892d96b5591622ec2a7ede00cdf892fb7ea2f595af5390a9f4e290ae` |

`cyclonedx-py environment` produced a runtime SBOM and `sha256sum -c SHA256SUMS` verified those files. The CI candidate workflow builds from the commit, independently installs on Python 3.10–3.12, and retains the wheel, sdist, checksum manifest, source commit, benchmark results, and SBOM from the Python 3.12 job. The report is excluded from the source archive so its checksum can be stored here.
