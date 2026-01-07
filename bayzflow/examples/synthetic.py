#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Synthetic latent generator
# -----------------------------
def generate_latent_sequence(T=200, T_switch=100, sigma_proc=0.05, drift=0.3, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    for t in range(1, T):
        if t < T_switch:
            z[t] = z[t-1] + rng.normal(0.0, sigma_proc)
        else:
            z[t] = z[t-1] + drift + rng.normal(0.0, sigma_proc)
    return z


def generate_posterior_samples(z_true, S=32, sigma_meas=0.1, seed=1):
    """Fake BNN posterior: S samples per time step around the true z."""
    rng = np.random.default_rng(seed)
    T = len(z_true)
    h = np.zeros((T, S))
    for t in range(T):
        h[t] = z_true[t] + rng.normal(0.0, sigma_meas, size=S)
    mu = h.mean(axis=1)
    var = h.var(axis=1, ddof=1)
    return h, mu, var


# -----------------------------
# 2. Kalman cognition
# -----------------------------
class KalmanCognition1D:
    def __init__(self, alpha_Q=1e-4, beta_v=1e-2, alpha_R=1e-4, beta_R=1.0):
        self.alpha_Q = alpha_Q
        self.beta_v = beta_v
        self.alpha_R = alpha_R
        self.beta_R = beta_R
        self.initialized = False

    def reset(self, z0=0.0, P0=1.0, mu0=None):
        self.z = z0
        self.P = P0
        self.mu_prev = mu0
        self.initialized = True

    def step(self, mu_t, var_t):
        if not self.initialized:
            self.reset(z0=mu_t, P0=1.0, mu0=mu_t)

        # latent speed
        v = abs(mu_t - self.mu_prev)
        self.mu_prev = mu_t

        # adaptive noises
        R_t = self.alpha_R + self.beta_R * var_t
        Q_t = self.alpha_Q + self.beta_v * v * v

        # predict
        z_pred = self.z
        P_pred = self.P + Q_t

        # innovation
        eps = mu_t - z_pred
        S_t = P_pred + R_t
        K_t = P_pred / S_t

        # update
        self.z = z_pred + K_t * eps
        self.P = (1.0 - K_t) * P_pred

        # belief
        belief = np.exp(-0.5 * eps * eps / S_t)

        return self.z, self.P, belief, eps, S_t


# -----------------------------
# 3. Particle-filter cognition
# -----------------------------
class ParticleFilterCognition1D:
    def __init__(self, N=200, alpha_Q=1e-4, beta_v=1e-2, alpha_R=1e-4, beta_R=1.0, ess_threshold=0.5):
        self.N = N
        self.alpha_Q = alpha_Q
        self.beta_v = beta_v
        self.alpha_R = alpha_R
        self.beta_R = beta_R
        self.ess_threshold = ess_threshold
        self.rng = np.random.default_rng(123)
        self.initialized = False

    def reset(self, z0=0.0, init_spread=0.1, mu0=None):
        self.particles = self.rng.normal(z0, init_spread, size=self.N)
        self.weights = np.ones(self.N) / self.N
        self.mu_prev = mu0
        self.initialized = True

    def _systematic_resample(self):
        """Systematic resampling."""
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
        self.weights = np.ones(N) / N

    def step(self, mu_t, var_t):
        if not self.initialized:
            self.reset(z0=mu_t, init_spread=np.sqrt(var_t + 1e-3), mu0=mu_t)

        # latent speed
        v = abs(mu_t - self.mu_prev)
        self.mu_prev = mu_t

        # adaptive noises
        R_t = self.alpha_R + self.beta_R * var_t
        Q_t = self.alpha_Q + self.beta_v * v * v

        # predict particles
        self.particles += self.rng.normal(0.0, np.sqrt(Q_t), size=self.N)

        # update weights based on measurement mu_t
        meas_var = var_t + R_t
        if meas_var <= 0:
            meas_var = 1e-6
        coeff = 1.0 / np.sqrt(2.0 * np.pi * meas_var)
        diff = mu_t - self.particles
        like = coeff * np.exp(-0.5 * diff * diff / meas_var)
        self.weights *= like
        sum_w = self.weights.sum()
        if sum_w == 0:
            # degeneracy fallback
            self.weights[:] = 1.0 / self.N
        else:
            self.weights /= sum_w

        # estimate
        z_hat = np.sum(self.weights * self.particles)
        var_pf = np.sum(self.weights * (self.particles - z_hat) ** 2)

        # ESS
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.ess_threshold * self.N:
            self._systematic_resample()
            ess = self.N

        # innovation + belief
        eps = mu_t - z_hat
        S_pf = var_t + R_t + var_pf
        belief = np.exp(-0.5 * eps * eps / S_pf)

        return z_hat, var_pf, belief, eps, S_pf, ess


# -----------------------------
# 4. Run experiment
# -----------------------------
def run_experiment(
    T=200,
    T_switch=100,
    sigma_proc=0.05,
    drift=0.3,
    S=32,
    sigma_meas=0.1,
):
    # True latent
    z_true = generate_latent_sequence(T, T_switch, sigma_proc, drift)
    # Fake BNN posterior samples
    h, mu, var = generate_posterior_samples(z_true, S, sigma_meas)

    # KF cognition
    kf = KalmanCognition1D(alpha_Q=1e-4, beta_v=1e-2, alpha_R=1e-4, beta_R=1.0)
    kf.reset(z0=mu[0], P0=1.0, mu0=mu[0])

    z_kf_hist = np.zeros(T)
    P_hist = np.zeros(T)
    belief_kf = np.zeros(T)
    eps_kf = np.zeros(T)

    # PF cognition
    pf = ParticleFilterCognition1D(N=300, alpha_Q=1e-4, beta_v=1e-2, alpha_R=1e-4, beta_R=1.0, ess_threshold=0.5)
    pf.reset(z0=mu[0], init_spread=np.sqrt(var[0] + 1e-3), mu0=mu[0])

    z_pf_hist = np.zeros(T)
    var_pf_hist = np.zeros(T)
    belief_pf = np.zeros(T)
    eps_pf = np.zeros(T)
    ess_hist = np.zeros(T)

    for t in range(T):
        # Kalman step
        z_kf, P, b_kf, e_kf, S_kf = kf.step(mu[t], var[t])
        z_kf_hist[t] = z_kf
        P_hist[t] = P
        belief_kf[t] = b_kf
        eps_kf[t] = e_kf

        # PF step
        z_pf, var_pf, b_pf, e_pf, S_pf, ess = pf.step(mu[t], var[t])
        z_pf_hist[t] = z_pf
        var_pf_hist[t] = var_pf
        belief_pf[t] = b_pf
        eps_pf[t] = e_pf
        ess_hist[t] = ess

    return {
        "z_true": z_true,
        "mu": mu,
        "var": var,
        "z_kf": z_kf_hist,
        "P": P_hist,
        "belief_kf": belief_kf,
        "eps_kf": eps_kf,
        "z_pf": z_pf_hist,
        "var_pf": var_pf_hist,
        "belief_pf": belief_pf,
        "eps_pf": eps_pf,
        "ess": ess_hist,
        "T_switch": T_switch,
    }


# -----------------------------
# 5. Plotting
# -----------------------------
def plot_results(results):
    z_true = results["z_true"]
    mu = results["mu"]
    z_kf = results["z_kf"]
    z_pf = results["z_pf"]
    belief_kf = results["belief_kf"]
    belief_pf = results["belief_pf"]
    T_switch = results["T_switch"]

    t = np.arange(len(z_true))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, z_true, label="True latent", linewidth=2)
    plt.plot(t, mu, label="Posterior mean (mu)", alpha=0.5)
    plt.plot(t, z_kf, label="KF estimate", linestyle="--")
    plt.plot(t, z_pf, label="PF estimate", linestyle=":")
    plt.axvline(T_switch, color="red", linestyle="--", label="Regime change")
    plt.legend()
    plt.title("Latent trajectories")

    plt.subplot(2, 1, 2)
    plt.plot(t, belief_kf, label="Belief KF")
    plt.plot(t, belief_pf, label="Belief PF")
    plt.axvline(T_switch, color="red", linestyle="--", label="Regime change")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.title("Belief over time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)
