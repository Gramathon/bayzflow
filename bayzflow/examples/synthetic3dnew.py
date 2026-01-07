#!/usr/bin/env python3
"""
3D latent cognition demo with belief divergence:
- Curved 3D latent trajectory with regime change
- Fake BNN posterior samples in 3D
- 3D Kalman + Particle Filter cognition
- STATIC: 3D latent + belief(KF/PF) + belief gap (PF - KF)
- ANIMATED: 3D latent + time-evolving belief + belief gap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


# ============================================================
# 1. 3D latent generator
# ============================================================

def rotation_matrix_z(theta: float) -> np.ndarray:
    """2D rotation in x-y plane (around z-axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def generate_latent_sequence_3d(
    T: int = 240,
    T_switch: int = 140,
    base_speed_a: float = 0.02,
    base_speed_b: float = 0.08,
    curve_rate_a: float = 0.01,
    curve_rate_b: float = 0.03,
    sigma_proc: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a 3D latent that slowly spirals in x-y with small z drift,
    then switches to a faster, sharper spiral at T_switch.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros((T, 3), dtype=float)

    dir0 = np.array([1.0, 0.0, 0.02])  # initial direction (slight z drift)

    for t in range(1, T):
        if t < T_switch:
            angle = curve_rate_a * t
            speed = base_speed_a
        else:
            angle = curve_rate_b * t
            speed = base_speed_b

        Rz = rotation_matrix_z(angle)
        drift = Rz @ dir0 * speed
        z[t] = z[t - 1] + drift + rng.normal(0.0, sigma_proc, size=3)

    return z


def generate_posterior_samples_3d(
    z_true: np.ndarray,
    S: int = 32,
    sigma_meas: float = 0.06,
    seed: int = 1,
):
    """
    Fake 3D BNN posterior samples around the true latent.
    z_true: (T, 3)
    Returns: h (T, S, 3), mu (T, 3), cov (T, 3, 3)
    """
    rng = np.random.default_rng(seed)
    T = z_true.shape[0]
    h = np.zeros((T, S, 3), dtype=float)
    for t in range(T):
        h[t] = z_true[t] + rng.normal(0.0, sigma_meas, size=(S, 3))

    mu = h.mean(axis=1)
    cov = np.zeros((T, 3, 3), dtype=float)
    for t in range(T):
        diffs = h[t] - mu[t]
        cov[t] = diffs.T @ diffs / max(1, (S - 1))

    return h, mu, cov


# ============================================================
# 2. 3D Kalman cognition
# ============================================================

class KalmanCognition3D:
    def __init__(self, alpha_Q=5e-4, beta_v=1e-1, alpha_R=1e-4, beta_R=1.0):
        """
        Q_t = alpha_Q + beta_v * ||Δμ||^2
        R_t = alpha_R + beta_R * (tr(cov_t) / 3)
        """
        self.alpha_Q = alpha_Q
        self.beta_v = beta_v
        self.alpha_R = alpha_R
        self.beta_R = beta_R
        self.initialized = False

    def reset(self, z0, P0=None, mu0=None):
        self.z = np.asarray(z0, dtype=float)
        if P0 is None:
            P0 = np.eye(3)
        self.P = P0
        self.mu_prev = np.asarray(mu0, dtype=float)
        self.initialized = True

    def step(self, mu_t, cov_t):
        if not self.initialized:
            self.reset(mu_t, P0=np.eye(3), mu0=mu_t)

        mu_t = np.asarray(mu_t)
        cov_t = np.asarray(cov_t)

        # latent speed
        v = np.linalg.norm(mu_t - self.mu_prev)
        self.mu_prev = mu_t

        # adaptive noises
        R_scalar = self.alpha_R + self.beta_R * np.trace(cov_t) / 3.0
        Q_scalar = self.alpha_Q + self.beta_v * v * v
        R_t = R_scalar * np.eye(3)
        Q_t = Q_scalar * np.eye(3)

        # predict (A = I)
        z_pred = self.z
        P_pred = self.P + Q_t

        # innovation
        eps = mu_t - z_pred
        S_t = P_pred + R_t
        S_inv = np.linalg.inv(S_t)
        K_t = P_pred @ S_inv

        # update
        self.z = z_pred + K_t @ eps
        self.P = (np.eye(3) - K_t) @ P_pred

        # Machine Belief (innovation likelihood)
        quad = eps.T @ S_inv @ eps
        belief = np.exp(-0.5 * quad)

        return self.z.copy(), self.P.copy(), belief, eps, S_t


# ============================================================
# 3. 3D Particle-filter cognition
# ============================================================

class ParticleFilterCognition3D:
    def __init__(
        self,
        N=1500,
        alpha_Q=5e-4,
        beta_v=1e-1,
        alpha_R=1e-4,
        beta_R=1.0,
        ess_threshold=0.3,
        jitter_after_resample=3e-3,
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

    def reset(self, z0, init_spread=0.15, mu0=None):
        self.particles = self.rng.normal(
            loc=z0, scale=init_spread, size=(self.N, 3)
        )
        self.weights = np.ones(self.N, dtype=float) / self.N
        self.mu_prev = np.asarray(mu0, dtype=float)
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

        # jitter to prevent collapse
        self.particles += self.rng.normal(
            0.0, np.sqrt(self.jitter_after_resample), size=self.particles.shape
        )

    def step(self, mu_t, cov_t):
        if not self.initialized:
            self.reset(mu_t, init_spread=np.sqrt(np.trace(cov_t) / 3 + 1e-3), mu0=mu_t)

        mu_t = np.asarray(mu_t)
        cov_t = np.asarray(cov_t)

        # latent speed
        v = np.linalg.norm(mu_t - self.mu_prev)
        self.mu_prev = mu_t

        # adaptive noises
        R_scalar = self.alpha_R + self.beta_R * np.trace(cov_t) / 3.0
        Q_scalar = self.alpha_Q + self.beta_v * v * v
        R_t = R_scalar * np.eye(3)
        Q_t = Q_scalar * np.eye(3)

        # predict
        self.particles += self.rng.normal(
            0.0, np.sqrt(Q_scalar), size=self.particles.shape
        )

        # measurement likelihood (3D Gaussian)
        meas_cov = cov_t + R_t
        meas_cov_inv = np.linalg.inv(meas_cov)
        det_meas = np.linalg.det(meas_cov)
        if det_meas <= 0:
            det_meas = 1e-9
        norm_const = 1.0 / (np.power(2.0 * np.pi, 1.5) * np.sqrt(det_meas))

        diffs = self.particles - mu_t  # (N,3)
        quad = np.einsum("ni,ij,nj->n", diffs, meas_cov_inv, diffs)
        like = norm_const * np.exp(-0.5 * quad)

        self.weights *= like
        s = self.weights.sum()
        if s <= 0.0:
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

        # ESS
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.ess_threshold * self.N:
            self._systematic_resample()
            ess = self.N

        # Machine Belief
        eps = mu_t - z_hat
        S_pf = meas_cov + var_pf
        S_inv = np.linalg.inv(S_pf)
        quad_eps = eps.T @ S_inv @ eps
        belief = np.exp(-0.5 * quad_eps)

        return z_hat.copy(), var_pf.copy(), belief, eps, S_pf, ess


# ============================================================
# 4. Run experiment
# ============================================================

def run_experiment_3d(
    T: int = 240,
    T_switch: int = 140,
    sigma_proc: float = 0.01,
    S: int = 32,
    sigma_meas: float = 0.06,
):
    z_true = generate_latent_sequence_3d(
        T=T,
        T_switch=T_switch,
        sigma_proc=sigma_proc,
    )
    h, mu, cov = generate_posterior_samples_3d(
        z_true, S=S, sigma_meas=sigma_meas
    )

    # Kalman
    kf = KalmanCognition3D()
    kf.reset(mu[0], P0=np.eye(3), mu0=mu[0])

    z_kf = np.zeros_like(z_true)
    belief_kf = np.zeros(T)

    # PF
    pf = ParticleFilterCognition3D()
    pf.reset(mu[0], init_spread=0.2, mu0=mu[0])

    z_pf = np.zeros_like(z_true)
    belief_pf = np.zeros(T)
    particles_history = np.zeros((T, pf.N, 3), dtype=float)

    for t in range(T):
        zk, _, bk, _, _ = kf.step(mu[t], cov[t])
        z_kf[t] = zk
        belief_kf[t] = bk

        zp, _, bp, _, _, _ = pf.step(mu[t], cov[t])
        z_pf[t] = zp
        belief_pf[t] = bp
        particles_history[t] = pf.particles

    # Belief divergence (PF vs KF)
    belief_gap = belief_pf - belief_kf

    return {
        "z_true": z_true,
        "mu": mu,
        "z_kf": z_kf,
        "z_pf": z_pf,
        "belief_kf": belief_kf,
        "belief_pf": belief_pf,
        "belief_gap": belief_gap,
        "particles": particles_history,
        "T_switch": T_switch,
    }


# ============================================================
# 5. STATIC: 3D + belief + gap
# ============================================================

def plot_static_3d(results):
    z_true = results["z_true"]
    mu = results["mu"]
    z_kf = results["z_kf"]
    z_pf = results["z_pf"]
    belief_kf = results["belief_kf"]
    belief_pf = results["belief_pf"]
    belief_gap = results["belief_gap"]
    T_switch = results["T_switch"]

    T = len(z_true)
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

    # 3D latent (left, spans rows)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(
        z_true[:, 0], z_true[:, 1], z_true[:, 2],
        color="black", lw=2.0, label="True latent"
    )
    ax3d.plot(
        mu[:, 0], mu[:, 1], mu[:, 2],
        color="orange", lw=1.5, alpha=0.6, label="Posterior mean"
    )
    ax3d.plot(
        z_kf[:, 0], z_kf[:, 1], z_kf[:, 2],
        color="green", lw=1.5, ls="--", label="KF estimate"
    )
    ax3d.plot(
        z_pf[:, 0], z_pf[:, 1], z_pf[:, 2],
        color="red", lw=1.5, ls=":", label="PF estimate"
    )

    # PF particles at regime change (for visual)
    ax3d.scatter(
        results["particles"][T_switch, :, 0],
        results["particles"][T_switch, :, 1],
        results["particles"][T_switch, :, 2],
        s=4, alpha=0.12, c="tab:blue", label="PF particles"
    )

    ax3d.scatter(
        [z_true[T_switch, 0]],
        [z_true[T_switch, 1]],
        [z_true[T_switch, 2]],
        c="red", marker="x", s=80, label="Regime change"
    )

    ax3d.set_xlabel("Latent dim 1")
    ax3d.set_ylabel("Latent dim 2")
    ax3d.set_zlabel("Latent dim 3")
    ax3d.set_title("3D latent cognition (PF particles + KF/PF tracks)")
    ax3d.legend(loc="upper left")
    ax3d.grid(True, linestyle=":", linewidth=0.5)
    ax3d.view_init(elev=25, azim=40)

    # Belief plot (top-right)
    ax_belief = fig.add_subplot(gs[0, 1])
    ax_belief.plot(t_axis, belief_kf, label="Belief KF", color="blue")
    ax_belief.plot(t_axis, belief_pf, label="Belief PF", color="orange")
    ax_belief.axvline(T_switch, color="red", ls="--", label="Regime change")
    ax_belief.set_xlim(0, T)
    ax_belief.set_ylim(0, 1.05)
    ax_belief.set_ylabel("Belief")
    ax_belief.set_title("Machine Belief over time")
    ax_belief.legend(loc="upper right")

    # Belief gap (bottom-right)
    ax_gap = fig.add_subplot(gs[1, 1])
    ax_gap.axhline(0.0, color="gray", lw=1, ls=":")
    ax_gap.plot(t_axis, belief_gap, color="purple", label="Belief gap (PF - KF)")
    ax_gap.axvline(T_switch, color="red", ls="--", label="Regime change")
    ax_gap.set_xlim(0, T)
    ax_gap.set_xlabel("Time")
    ax_gap.set_ylabel("ΔBelief")
    ax_gap.set_title("Belief divergence")
    ax_gap.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


# ============================================================
# 6. ANIMATION: 3D + belief + gap
# ============================================================

def animate_3d(results, save_path=None, fps=20):
    z_true = results["z_true"]
    mu = results["mu"]
    z_kf = results["z_kf"]
    z_pf = results["z_pf"]
    belief_kf = results["belief_kf"]
    belief_pf = results["belief_pf"]
    belief_gap = results["belief_gap"]
    particles_hist = results["particles"]
    T_switch = results["T_switch"]

    T, N, _ = particles_hist.shape
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_belief = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, 1])

    # Full true latent for context
    ax3d.plot(
        z_true[:, 0], z_true[:, 1], z_true[:, 2],
        color="black", lw=1.5, label="True latent"
    )

    scat = ax3d.scatter(
        particles_hist[0, :, 0],
        particles_hist[0, :, 1],
        particles_hist[0, :, 2],
        s=3,
        alpha=0.25,
        label="PF particles",
        c="tab:blue",
    )

    line_mu, = ax3d.plot(
        mu[:1, 0], mu[:1, 1], mu[:1, 2],
        color="orange", lw=2, label="Posterior mean"
    )
    line_kf, = ax3d.plot(
        z_kf[:1, 0], z_kf[:1, 1], z_kf[:1, 2],
        color="green", lw=2, ls="--", label="KF estimate"
    )
    line_pf, = ax3d.plot(
        z_pf[:1, 0], z_pf[:1, 1], z_pf[:1, 2],
        color="red", lw=2, ls=":", label="PF estimate"
    )

    ax3d.scatter(
        [z_true[T_switch, 0]],
        [z_true[T_switch, 1]],
        [z_true[T_switch, 2]],
        c="red", marker="x", s=80, label="Regime change"
    )
    ax3d.set_xlabel("Latent dim 1")
    ax3d.set_ylabel("Latent dim 2")
    ax3d.set_zlabel("Latent dim 3")
    ax3d.set_title("3D latent cognition (PF particles + KF/PF tracks)")
    ax3d.legend(loc="upper left")
    ax3d.grid(True, linestyle=":", linewidth=0.5)
    ax3d.view_init(elev=25, azim=40)

    # Belief plot
    ax_belief.set_xlim(0, T)
    ax_belief.set_ylim(0, 1.05)
    line_bk, = ax_belief.plot([], [], label="Belief KF", color="blue")
    line_bp, = ax_belief.plot([], [], label="Belief PF", color="orange")
    ax_belief.axvline(T_switch, color="red", ls="--", label="Regime change")
    ax_belief.set_ylabel("Belief")
    ax_belief.set_title("Machine Belief over time")
    ax_belief.legend(loc="upper right")

    # Belief gap plot
    ax_gap.set_xlim(0, T)
    # gap is in [-1, 1], but mostly [-0.5, 0.5]
    ax_gap.set_ylim(-1.0, 1.0)
    ax_gap.axhline(0.0, color="gray", lw=1, ls=":")
    line_gap, = ax_gap.plot([], [], color="purple", label="Belief gap (PF - KF)")
    ax_gap.axvline(T_switch, color="red", ls="--", label="Regime change")
    ax_gap.set_xlabel("Time")
    ax_gap.set_ylabel("ΔBelief")
    ax_gap.set_title("Belief divergence")
    ax_gap.legend(loc="upper right")

    fig.tight_layout()

    def update(frame):
        # update particles
        scat._offsets3d = (
            particles_hist[frame, :, 0],
            particles_hist[frame, :, 1],
            particles_hist[frame, :, 2],
        )

        # update trajectories
        line_mu.set_data(mu[: frame + 1, 0], mu[: frame + 1, 1])
        line_mu.set_3d_properties(mu[: frame + 1, 2])

        line_kf.set_data(z_kf[: frame + 1, 0], z_kf[: frame + 1, 1])
        line_kf.set_3d_properties(z_kf[: frame + 1, 2])

        line_pf.set_data(z_pf[: frame + 1, 0], z_pf[: frame + 1, 1])
        line_pf.set_3d_properties(z_pf[: frame + 1, 2])

        # rotate camera slowly
        ax3d.view_init(elev=25, azim=40 + 0.4 * frame)

        # update belief curves
        line_bk.set_data(t_axis[: frame + 1], belief_kf[: frame + 1])
        line_bp.set_data(t_axis[: frame + 1], belief_pf[: frame + 1])

        # update belief gap
        line_gap.set_data(t_axis[: frame + 1], belief_gap[: frame + 1])

        return scat, line_mu, line_kf, line_pf, line_bk, line_bp, line_gap

    anim = FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False
    )

    if save_path is not None:
        if save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps)
        elif save_path.endswith(".gif"):
            anim.save(save_path, writer="imagemagick", fps=fps)

    return anim


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    results = run_experiment_3d()
    plot_static_3d(results)

    # Interactive animation:
    anim = animate_3d(results)
    plt.show()

    # To save instead of showing:
    # animate_3d(results, save_path="latent_cognition_3d_divergence.mp4", fps=20)
