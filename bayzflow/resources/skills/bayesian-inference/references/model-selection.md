# Model and layer selection

Call `list_candidates()` and choose a subset by stable module-name patterns. Current candidate wrappers cover Linear, Conv1d/2d/3d, and ConvTranspose3d modules. Normalization, ConvTranspose1d/2d and arbitrary composite modules are not candidates. Explicitly selecting an unsupported module raises an error. Check tensor shapes, latent parameter count and device behavior on a tiny batch before training.
