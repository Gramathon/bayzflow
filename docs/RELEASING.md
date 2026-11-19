# Releasing Bayzflow

1. Confirm that the version in the release tag, `bayzflow/_version.py`, wheel metadata, changelog, CI install checks, and example pin agree.
2. Review the test matrix, clean CPU installs, scientific benchmark, MCP checks, security scans, and candidate artifact manifests from a successful push run on `main`.
3. After release authorization, push a lightweight version tag to the exact GitHub verified commit that passed CI. Ensure the repository has immutable releases enabled.

```bash
git tag v0.2.1 <verified-main-commit>
git push origin refs/tags/v0.2.1
```

The tag push starts **Publish validated release**. It checks the GitHub verified source commit, verifies the attestations and SHA256 checksums of the candidate from that commit's main CI run, and compares the uploaded draft assets byte for byte before publication. It uses the existing CI artifacts. If the tag was pushed before CI finished, rerun the release workflow after CI passes.

Merging the release PR runs CI on `main`; it does not publish a release. After that exact `main` commit has a successful CI run, push its authorized version tag to start **Publish validated release**. After publication, download the assets, run `sha256sum -c SHA256SUMS`, verify with `gh release verify v0.2.1`, and install the published wheel in a fresh environment. Update the GitHub Release installation links in `README.md` and `INSTALL.md` to v0.2.1 after the release exists. Record the results with Issue #1.
