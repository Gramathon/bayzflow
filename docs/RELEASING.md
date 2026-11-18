# Releasing Bayzflow

This document describes the required release procedure; it does not authorize publication. Do not create a tag or publish a release until maintainers authorize it and every gate in `docs/engineering/RELEASE_READINESS.md` is green.

1. Start from a clean reviewed commit on the release branch. Confirm `bayzflow/_version.py`, wheel metadata and changelog use one version.
2. Run the full supported Python test matrix, lint/type/security jobs, scientific benchmark, MCP protocol tests and installed-skill checks.
3. Build the exact wheel and sdist once with `uv build` (or the CI build frontend). Install the wheel in a fresh environment outside the checkout and run the documented CLI, MCP and skill checks.
4. Generate SHA256 sums for those exact files. Retain environment, commit, lockfile, workflow run and artifact records alongside them.
5. Review release notes, security status, SBOM/provenance and limitations. Only then create an authorized signed version tag and publish the already validated artifacts without rebuilding them.
6. Verify the published artifact hashes and installation from the release URL.

On a green main push, CI uploads `bayzflow-candidate` and attests the wheel, sdist and runtime SBOM. After review and release authorization, use the **Publish validated release** workflow on main with the signed tag and successful main push CI run ID. Configure the GitHub `release` environment with the repository's release reviewers before dispatch. The workflow verifies that the tag is signed, the commit matches the run, and the downloaded files match the checksum manifest and attestations. It publishes those files directly.

After publication, download the files into an empty directory, run `sha256sum -c SHA256SUMS`, verify their GitHub attestations, and install the wheel in a fresh environment. Record the release URL and results on Issue #1.
