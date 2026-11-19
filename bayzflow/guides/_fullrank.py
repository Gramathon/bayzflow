"""Budget for Pyro guides that allocate dense covariance factors."""

from pyro.infer.autoguide import AutoMultivariateNormal

MAX_FULLRANK_LATENTS = 512


def check_fullrank_dimension(dimension: int) -> None:
    if dimension > MAX_FULLRANK_LATENTS:
        raise ValueError(
            f"auto_fullrank supports at most {MAX_FULLRANK_LATENTS} latent coordinates; "
            f"found {dimension}. Select fewer layers or use auto_lowrank or auto_diag."
        )


class BoundedFullRankGuide(AutoMultivariateNormal):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Keep the Pyro parameter namespace used by existing checkpoints.
        self._pyro_name = AutoMultivariateNormal.__name__

    def _init_loc(self):
        # Pyro has traced the unconstrained latents but has yet to allocate scale_tril.
        try:
            check_fullrank_dimension(self.latent_dim)
        except ValueError:
            self.prototype_trace = None
            raise
        return super()._init_loc()
