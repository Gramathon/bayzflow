# Bayzflow 0.2.0 release readiness

Candidate validation covers CPU inference, all five guides, clean wheel installation, an EXPCON frozen execution, and controlled MCP tools over stdio. The final distribution and release approval are separate gates.

| Gate | Evidence |
|---|---|
| Python engine, public API and custom guide | 22 passing tests against the installed Python 3.12 CPU wheel |
| Analytical check | Three seeds × five guides, finite samples and measured deviations from the exact posterior |
| MCP jobs and denial controls | Official stdio client, fit and sample, both measurements, artifact checks, cancellation and rejection tests |
| EXPCON handoff | Verified frozen run with a passing preregistered threshold |
| Packaging | Wheel and sdist, strict metadata check, skill installation, SHA256 manifest and runtime SBOM |
| CI and security | Review the checks for the current [draft PR](https://github.com/Gramathon/bayzflow/pull/2), including the installed CPU matrix and security gates |
| Release | Requires review, source attestation on main, an authorized tag on a GitHub verified commit, publication of the recorded CI artifacts, and installation from the GitHub Release |

The Pi interpretation attempt hit missing provider credentials. GPU and HTTP operations are outside the validated scope. Older tracked research runs and separate user files in the original checkout are outside the production test suite for this PR.

Keep Issue #1 open through validated publication. The issue's release criterion is fulfilled only after checking the downloaded GitHub Release files against the manifest and installing the released wheel.
