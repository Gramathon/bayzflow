# Posterior sampling

Use `train_posterior(..., save_dir=...)` to fit and persist a posterior. Use `predict(..., num_samples=N, return_samples=True)` for a bounded sample count. The checkpoint contains model tensors, variational tensors, guide metadata and declarative constraint metadata. Treat checkpoint directories as trusted artifacts and retain their checksums and input/configuration identities.
