# SVI training regimes

Selected supported layers become native Pyro weight/bias sample sites. Their original values center independent Gaussian priors; variational guide parameters are optimized by Adam/Trace_ELBO. Unselected trainable parameters are point estimates, not posterior random variables.

`BayzFlow(..., deterministic_update_policy='joint')` preserves the existing default: unselected parameters with requires_grad=True may update. Existing manually frozen parameters remain frozen. `deterministic_update_policy='freeze'` freezes remaining parameters when building the Bayesian model and validates that none were subsequently reenabled before setup or fitting. This operates on the supplied model, which layer wrapping already modifies in place. Prior buffers are not optimized. The factory methods `from_model_and_data` and `auto` accept the same policy.

To freeze a pretrained backbone, load pretrained weights, select only the head, and choose `freeze`. For joint point estimation choose `joint`. To represent every intended supported layer through the posterior, select every such layer and inspect the remaining deterministic inventory. This does not make unsupported components Bayesian. The executable examples are in `bayzflow/examples/svi_training_controls.py`:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m bayzflow.examples.svi_training_controls
```

## Training lifecycle

```python
bf.setup_svi(loss, predict, lr=0.02)
bf.fit(loader, num_epochs=50)
bf.fit(loader, num_epochs=50, continue_optimizer=True)
bf.reset_optimizer()  # retain guide parameters, empty Adam moments
bf.setup_svi(loss, predict, lr=0.02, restart_posterior=True)
```

The final operation clears this engine's variational store and creates a fresh guide under the configured initialization policy. Set the Pyro seed explicitly for reproducible initialization. It retains the current model and prior buffers; restarting a joint fit does not restore the original deterministic backbone. Reconstruct a fresh model for that operation.

The default `continue_optimizer=False` preserves the historically observed fresh-Adam behaviour at fit boundaries. `True` retains parameter identities and Adam state within this process. Ten uninterrupted updates and two five-update calls match bitwise under fixed seeds and batches, including joint point parameters. Exact continuation also requires the same RNG stream, data order, device and dtype; diagnostics preserve the RNG state once the guide is initialized. History accumulates across fit calls and `svi_steps` counts batches, not epochs.

Safe posterior checkpoints preserve model and guide tensors, not Adam/RNG/data-loader state. Continuing an optimizer directly after load raises a clear error; `fit(..., continue_optimizer=False)` starts a fresh optimizer. No across-process exact-continuation claim is made. Checkpoints record the training policy; loading with an incompatible policy fails. Legacy checkpoints imply the legacy joint policy.

## Inspection

`bf.diagnostics(batch)` returns JSON-compatible parameter inventory, prior buffers, guide family, latent dimensions and site offsets, optimizer configuration/state step counts, ELBO history, deterministic update norms and guide update norms. A null guide delta means the parameter was newly initialized in that call, not that it did not update. Deterministic deltas describe the most recent fit. Guide covariance remains a tensor-valued representation expressed as serializable arrays. Dense diagnostic output is limited to 256 latent dimensions by default (hard maximum 2048).

`bf.posterior_samples(..., site_names=[...])` followed by `bf.posterior_summary(samples)` gives bounded sitewise empirical native-weight moments. Preserve joint draw indices and explicitly concatenate matching sites to calculate cross-layer empirical blocks; sitewise summaries do not provide cross-layer covariance. Neither this covariance nor guide covariance is activation covariance or predictive covariance. Nonidentity support transforms require transforming samples; packed unconstrained covariance cannot simply be relabelled as native-weight covariance. Eigenvectors have arbitrary signs and unstable orientation in degenerate subspaces.

The generic loss callback owns observation distribution, noise, reduction and minibatch scaling. Full-data summed Gaussian NLL targets the stated Gaussian regression posterior; a mean loss changes that target. ELBO stabilization alone does not establish posterior accuracy. Coverage requires appropriate held-out validation; no output is automatically labelled calibrated.

Model train/eval mode remains caller-controlled. Exact continuation also requires retaining that mode; dropout and normalization behaviour can change the objective if mode changes between segments. Diagnostics check latent size before initializing a fresh full-rank covariance and reject dimensions over the requested bound.

Recursive calls that return to engine A while engine B owns the global Pyro store are explicitly rejected. Sequential and concurrent engine operations remain isolated; this guard prevents callbacks from accidentally using another experiment's state.
