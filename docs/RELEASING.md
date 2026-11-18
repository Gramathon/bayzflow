# Releasing Bayzflow

1. Confirm that the version in the release tag, `bayzflow/_version.py`, wheel metadata, and changelog agree.
2. Review the test matrix, clean CPU installs, scientific benchmark, MCP checks, security scans, and candidate artifact manifests from a successful push run on `main`.
3. After release authorization, push a lightweight version tag to the exact GitHub verified commit that passed CI. Ensure the repository has immutable releases enabled.

```bash
git tag v0.2.0 <verified-main-commit>
git push origin refs/tags/v0.2.0
```

The tag push starts **Publish validated release**. It checks the GitHub verified source commit, verifies the attestations and SHA256 checksums of the candidate from that commit's main CI run, and compares the uploaded draft assets byte for byte before publication. It uses the existing CI artifacts. If the tag was pushed before CI finished, rerun the release workflow after CI passes.

After publication, download the assets, run `sha256sum -c SHA256SUMS`, verify the release with `gh release verify v0.2.0`, and install the published wheel in a fresh environment. Record the results with Issue #1.
