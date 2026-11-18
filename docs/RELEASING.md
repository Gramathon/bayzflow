# Releasing Bayzflow

This document describes the required release procedure; it does not authorize publication. Do not create a tag or publish a release until maintainers authorize it and every gate in `docs/engineering/RELEASE_READINESS.md` is green.

1. Start from a clean reviewed commit on the release branch. Confirm `bayzflow/_version.py`, wheel metadata and changelog use one version.
2. Run the full supported Python test matrix, lint/type/security jobs, scientific benchmark, MCP protocol tests and installed-skill checks.
3. Build the exact wheel and sdist once with `uv build` (or the CI build frontend). Install the wheel in a fresh environment outside the checkout and run the documented CLI, MCP and skill checks.
4. Generate SHA256 sums for those exact files. Retain environment, commit, lockfile, workflow run and artifact records alongside them.
5. Review release notes, security status, SBOM/provenance and limitations. Authorize the release workflow to sign the tag and publish the validated artifacts.
6. Verify the published artifact hashes and installation from the release URL.

On a green main push, CI uploads `bayzflow-candidate` and attests the wheel, sdist and runtime SBOM. Enable immutable releases on the repository and configure the `release` environment with the repository's release reviewers. Dispatch **Publish validated release** on main with the version and successful main CI run ID. For example:

```bash
gh workflow run release.yml --repo Gramathon/bayzflow --ref main \
  -f tag=v0.2.0 -f run_id=44393039927
```

The workflow verifies the candidate checksums and build attestations against the commit selected by the run. GitHub Actions OIDC and Sigstore sign the Git tag, binding the source commit, run and checksum manifest. The workflow verifies the signature against the main release workflow identity before pushing the tag. It uploads the verified files to a draft, downloads them to compare their contents, and publishes the release after those checks pass. Rerun with the same inputs to resume an incomplete draft.

After publication, download the files into an empty directory, run `sha256sum -c SHA256SUMS`, verify the GitHub release and build attestations, and install the wheel in a fresh environment. Record the release URL and results on Issue #1.

```bash
gitsign verify-tag v0.2.0 \
  --certificate-identity https://github.com/Gramathon/bayzflow/.github/workflows/release.yml@refs/heads/main \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```
