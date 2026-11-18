# Guide selection

Use `auto_normal`, `auto_diag`, `auto_lowrank`, or `auto_fullrank`. Inspect accepted arguments with `bayzflow guide inspect ID`. `auto_lowrank` accepts a positive integer `rank`; `auto_fullrank` uses dense covariance and can require quadratic memory in the number of latent parameters. Guide compatibility depends on the model's latent sites and Pyro version.
