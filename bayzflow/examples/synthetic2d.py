#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# 1. Synthetic 2D latent
# =========================

def generate_latent_sequence_2d(
    T=200,
    T_switch=100,
    sigma_proc=0.05,
    speed_a=0.01,
    speed_b=0.1,
    angle_a=np.deg2rad(5),
    angle_b=np.deg2rad(30),
    seed=0,
):
    """
    2D latent that slowly drifts in one direction, then switches
    to a faster drift in a different direction at T_switch.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros((T, 2))

    # unit direction vectors
    dir_a = np.array([np.cos(angle_a), np.sin(angle_a)]) * speed_a
    dir_b = np.array([np.cos(angle_b), np.sin(angle_b)]) * speed_b

    for t in range(1, T):
        if t < T_switch:
            drift = dir_a
        else:
            drift = dir_b
        z[t] = z[t - 1] + drift + rng.normal(0.0, sigma_proc, size=2)

    return z


def generate_posterior_samples_2d(z_true, S=32, sigma_meas=0.1, seed=1):
    """
    Fake BNN posterior samples around the true 2D latent.
    """
    rng = np.random.default_rng(seed)
    T = len(z_true)
    h = np.zeros((T, S, 2))
    for t in range(T):
        h[t] = z_true[t] + rng.normal(0.0, sigma_meas, size=(S, 2))
    mu = h.mean(axis=1)  # (T,2)
    cov = np.zeros((T, 2, 2))
    for t in range(T):
        diffs = h[t] - mu[t]
        cov[t] = diffs.T @ diffs / max(1, (S - 1))
    return h, mu, cov


# =========================
# 2. Kalman cognition (2D)
# =========================

class KalmanCognition2D:
    def __init__(self, alpha_Q=1e-4, beta_v=5e-2, alpha_R=1e-4, beta_R=1.0):
        self.alpha_Q = alpha_Q
        self.beta_v = beta_v
        self.alpha_R = alpha_R
        self.beta_R = beta_R
        self.initialized = False

    def reset(self, z0, P0=None, mu0=None):
        self.z = np.asarray(z0, dtype=float)
        if P0 is None:
            P0 = np.eye(2)
        self.P = P0
        self.mu_prev = np.asarray(mu0, dtype=float)
        self.initialized = True

    def step(self, mu_t, cov_t):
        if not self.initialized:
            self.reset(mu_t, P0=np.eye(2), mu0=mu_t)

        mu_t = np.asarray(mu_t)
        cov_t = np.asarray(cov_t)

        # latent speed (2-norm of posterior mean change)
        v = np.linalg.norm(mu_t - self.mu_prev)
        self.mu_prev = mu_t

        # adaptive noises (scalar -> isotropic)
        R_scalar = self.alpha_R + self.beta_R * np.trace(cov_t) / 2.0
        Q_scalar = self.alpha_Q + self.beta_v * v * v

        R_t = R_scalar * np.eye(2)
        Q_t = Q_scalar * np.eye(2)

        # predict (A = I)
        z_pred = self.z
        P_pred = self.P + Q_t

        # innovation
        eps = mu_t - z_pred  # (2,)
        S_t = P_pred + R_t   # (2,2)
        K_t = P_pred @ np.linalg.inv(S_t)

        # update
        self.z = z_pred + K_t @ eps
        self.P = (np.eye(2) - K_t) @ P_pred

        # belief (innovation likelihood under Gaussian)
        # eps^T S^{-1} eps
        quad = eps.T @ np.linalg.inv(S_t) @ eps
        belief = np.exp(-0.5 * quad)

        return self.z.copy(), self.P.copy(), belief, eps, S_t


# =========================
# 3. Particle-filter cognition (2D)
# =========================

class ParticleFilterCognition2D:
    def __init__(
        self,
        N=800,
        alpha_Q=1e-4,
        beta_v=5e-2,
        alpha_R=1e-4,
        beta_R=1.0,
        ess_threshold=0.3,
        jitter_after_resample=1e-3,
        seed=123,
    ):
        self.N = N
        self.alpha_Q = alpha_Q
        self.beta_v = beta_v
        self.alpha_R = alpha_R
        self.beta_R = beta_R
        self.ess_threshold = ess_threshold
        self.jitter_after_resample = jitter_after_resample
        self.rng = np.random.default_rng(seed)
        self.initialized = False

    def reset(self, z0, init_spread=0.2, mu0=None):
        self.particles = self.rng.normal(
            loc=z0, scale=init_spread, size=(self.N, 2)
        )
        self.weights = np.ones(self.N) / self.N
        self.mu_prev = np.array(mu0, dtype=float)
        self.initialized = True

    def _systematic_resample(self):
        N = self.N
        positions = (self.rng.random() + np.arange(N)) / N
        cumulative_sum = np.cumsum(self.weights)
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights[:] = 1.0 / N

        # small jitter to prevent collapse
        self.particles += self.rng.normal(
            0.0, np.sqrt(self.jitter_after_resample), size=self.particles.shape
        )

    def step(self, mu_t, cov_t):
        if not self.initialized:
            self.reset(mu_t, init_spread=np.sqrt(np.trace(cov_t) / 2 + 1e-3), mu0=mu_t)

        mu_t = np.asarray(mu_t)
        cov_t = np.asarray(cov_t)

        # latent speed
        v = np.linalg.norm(mu_t - self.mu_prev)
        self.mu_prev = mu_t

        # adaptive noises
        R_scalar = self.alpha_R + self.beta_R * np.trace(cov_t) / 2.0
        Q_scalar = self.alpha_Q + self.beta_v * v * v

        R_t = R_scalar * np.eye(2)
        Q_t = Q_scalar * np.eye(2)

        # predict particles
        self.particles += self.rng.normal(
            0.0, np.sqrt(Q_scalar), size=self.particles.shape
        )

        # update weights using 2D Gaussian likelihood
        meas_cov = cov_t + R_t
        inv_meas_cov = np.linalg.inv(meas_cov)
        det_meas_cov = np.linalg.det(meas_cov)
        if det_meas_cov <= 0:
            det_meas_cov = 1e-9
        norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_meas_cov))

        diffs = self.particles - mu_t  # (N,2)
        quad = np.einsum("ni,ij,nj->n", diffs, inv_meas_cov, diffs)
        like = norm_const * np.exp(-0.5 * quad)

        self.weights *= like
        s = self.weights.sum()
        if s == 0:
            self.weights[:] = 1.0 / self.N
        else:
            self.weights /= s

        # estimate
        z_hat = np.sum(self.weights[:, None] * self.particles, axis=0)
        centered = self.particles - z_hat
        var_pf = np.sum(
            self.weights[:, None, None]
            * np.einsum("ni,nj->nij", centered, centered),
            axis=0,
        )

        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.ess_threshold * self.N:
            self._systematic_resample()
            ess = self.N

        # belief
        eps = mu_t - z_hat
        S_pf = meas_cov + var_pf
        inv_S = np.linalg.inv(S_pf)
        quad_eps = eps.T @ inv_S @ eps
        belief = np.exp(-0.5 * quad_eps)

        return z_hat.copy(), var_pf.copy(), belief, eps, S_pf, ess


# =========================
# 4. Run experiment
# =========================

def run_experiment_2d(
    T=220,
    T_switch=120,
    sigma_proc=0.01,
    S=32,
    sigma_meas=0.08,
):
    z_true = generate_latent_sequence_2d(
        T=T,
        T_switch=T_switch,
        sigma_proc=sigma_proc,
    )
    h, mu, cov = generate_posterior_samples_2d(
        z_true, S=S, sigma_meas=sigma_meas
    )

    # Kalman
    kf = KalmanCognition2D(alpha_Q=1e-4, beta_v=5e-2, alpha_R=1e-4, beta_R=1.0)
    kf.reset(mu[0], P0=np.eye(2), mu0=mu[0])

    z_kf = np.zeros_like(z_true)
    belief_kf = np.zeros(T)

    # PF
    pf = ParticleFilterCognition2D(
        N=800,
        alpha_Q=1e-4,
        beta_v=5e-2,
        alpha_R=1e-4,
        beta_R=1.0,
        ess_threshold=0.3,
        jitter_after_resample=1e-3,
    )
    pf.reset(mu[0], init_spread=0.1, mu0=mu[0])

    z_pf = np.zeros_like(z_true)
    belief_pf = np.zeros(T)
    particles_history = np.zeros((T, pf.N, 2))

    for t in range(T):
        zk, Pk, bk, _, _ = kf.step(mu[t], cov[t])
        z_kf[t] = zk
        belief_kf[t] = bk

        zp, varp, bp, _, _, ess = pf.step(mu[t], cov[t])
        z_pf[t] = zp
        belief_pf[t] = bp
        particles_history[t] = pf.particles

    return {
        "z_true": z_true,
        "mu": mu,
        "z_kf": z_kf,
        "z_pf": z_pf,
        "belief_kf": belief_kf,
        "belief_pf": belief_pf,
        "particles": particles_history,
        "T_switch": T_switch,
    }


# =========================
# 5. Static plot
# =========================

def plot_static_2d(results):
    z_true = results["z_true"]
    mu = results["mu"]
    z_kf = results["z_kf"]
    z_pf = results["z_pf"]
    T_switch = results["T_switch"]

    plt.figure(figsize=(8, 4))
    plt.plot(z_true[:, 0], z_true[:, 1], label="True latent", lw=2)
    plt.plot(mu[:, 0], mu[:, 1], label="Posterior mean", alpha=0.6)
    plt.plot(z_kf[:, 0], z_kf[:, 1], "--", label="KF estimate")
    plt.plot(z_pf[:, 0], z_pf[:, 1], ":", label="PF estimate")
    plt.scatter(
        [z_true[T_switch, 0]],
        [z_true[T_switch, 1]],
        c="red",
        marker="x",
        s=80,
        label="Regime change",
    )
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.legend()
    plt.title("2D latent trajectories with regime change")
    plt.tight_layout()
    plt.show()


# =========================
# 6. Animation (2D latent + belief)
# =========================

def animate_2d(results, save_path=None, fps=20):
    z_true = results["z_true"]
    mu = results["mu"]
    z_kf = results["z_kf"]
    z_pf = results["z_pf"]
    belief_kf = results["belief_kf"]
    belief_pf = results["belief_pf"]
    particles_hist = results["particles"]
    T_switch = results["T_switch"]

    T, N, _ = particles_hist.shape
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_latent = fig.add_subplot(gs[:, 0])
    ax_belief = fig.add_subplot(gs[1, 1])

    # pre-plot full true latent for context
    ax_latent.plot(
        z_true[:, 0], z_true[:, 1], color="black", lw=1.5, label="True latent"
    )

    scat_particles = ax_latent.scatter(
        particles_hist[0, :, 0],
        particles_hist[0, :, 1],
        s=5,
        alpha=0.3,
        label="PF particles",
    )
    line_mu, = ax_latent.plot(
        mu[:1, 0], mu[:1, 1], color="orange", lw=2, label="Posterior mean"
    )
    line_kf, = ax_latent.plot(
        z_kf[:1, 0], z_kf[:1, 1], "--", color="green", lw=2, label="KF estimate"
    )
    line_pf, = ax_latent.plot(
        z_pf[:1, 0], z_pf[:1, 1], ":", color="red", lw=2, label="PF estimate"
    )

    ax_latent.scatter(
        [z_true[T_switch, 0]],
        [z_true[T_switch, 1]],
        c="red",
        marker="x",
        s=80,
        label="Regime change",
    )

    ax_latent.set_xlabel("Latent dim 1")
    ax_latent.set_ylabel("Latent dim 2")
    ax_latent.legend(loc="upper left")
    ax_latent.set_title("Latent space cognition")

    # belief plot
    ax_belief.set_xlim(0, T)
    ax_belief.set_ylim(0, 1.05)
    line_bk, = ax_belief.plot(
        [], [], label="Belief KF", color="blue"
    )
    line_bp, = ax_belief.plot(
        [], [], label="Belief PF", color="orange"
    )
    ax_belief.axvline(T_switch, color="red", ls="--", label="Regime change")
    ax_belief.legend()
    ax_belief.set_xlabel("Time")
    ax_belief.set_ylabel("Belief")
    ax_belief.set_title("Machine Belief over time")

    fig.tight_layout()

    def update(frame):
        # latent
        scat_particles.set_offsets(particles_hist[frame])
        line_mu.set_data(mu[: frame + 1, 0], mu[: frame + 1, 1])
        line_kf.set_data(z_kf[: frame + 1, 0], z_kf[: frame + 1, 1])
        line_pf.set_data(z_pf[: frame + 1, 0], z_pf[: frame + 1, 1])

        # belief
        line_bk.set_data(t_axis[: frame + 1], belief_kf[: frame + 1])
        line_bp.set_data(t_axis[: frame + 1], belief_pf[: frame + 1])
        return (
            scat_particles,
            line_mu,
            line_kf,
            line_pf,
            line_bk,
            line_bp,
        )

    anim = FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False
    )

    if save_path is not None:
        # Choose mp4 or gif depending on extension
        if save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        elif save_path.endswith(".gif"):
            anim.save(save_path, writer="imagemagick", fps=fps)

    return anim


if __name__ == "__main__":
    results = run_experiment_2d()
    plot_static_2d(results)
    # For interactive use, just call animate_2d(results)
    # To save:
    # anim = animate_2d(results, save_path="latent_cognition_2d.mp4", fps=20)
    anim = animate_2d(results)
    plt.show()
