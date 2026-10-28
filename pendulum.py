"""
Double pendulum simulation — Lagrangian mechanics, RK4 integration,
chaos visualisation and Lyapunov exponent estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass

G = 9.81

# ── Equations of motion (Lagrangian) ─────────────────────────────────────────

def derivatives(state: np.ndarray, m1: float, m2: float,
                L1: float, L2: float) -> np.ndarray:
    th1, w1, th2, w2 = state
    delta = th2 - th1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    dth1 = w1
    dth2 = w2

    dw1 = (
        m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta)
        + m2 * G * np.sin(th2) * np.cos(delta)
        + m2 * L2 * w2**2 * np.sin(delta)
        - (m1 + m2) * G * np.sin(th1)
    ) / den1

    dw2 = (
        -m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * G * np.sin(th1) * np.cos(delta)
        - (m1 + m2) * L1 * w1**2 * np.sin(delta)
        - (m1 + m2) * G * np.sin(th2)
    ) / den2

    return np.array([dth1, dw1, dth2, dw2])

# ── RK4 integrator ───────────────────────────────────────────────────────────

def rk4_step(state: np.ndarray, dt: float, m1: float, m2: float,
             L1: float, L2: float) -> np.ndarray:
    k1 = derivatives(state, m1, m2, L1, L2)
    k2 = derivatives(state + 0.5 * dt * k1, m1, m2, L1, L2)
    k3 = derivatives(state + 0.5 * dt * k2, m1, m2, L1, L2)
    k4 = derivatives(state + dt * k3, m1, m2, L1, L2)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    t: np.ndarray
    th1: np.ndarray
    th2: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    energy: np.ndarray


def simulate(th1_0: float, th2_0: float, w1_0: float = 0, w2_0: float = 0,
             m1: float = 1, m2: float = 1, L1: float = 1, L2: float = 1,
             T: float = 30, dt: float = 0.005) -> SimResult:
    n_steps = int(T / dt)
    state = np.array([th1_0, w1_0, th2_0, w2_0], dtype=float)

    t_arr = np.zeros(n_steps)
    th1_arr = np.zeros(n_steps)
    th2_arr = np.zeros(n_steps)
    w1_arr = np.zeros(n_steps)
    w2_arr = np.zeros(n_steps)

    for i in range(n_steps):
        t_arr[i] = i * dt
        th1_arr[i] = state[0]
        th2_arr[i] = state[2]
        w1_arr[i] = state[1]
        w2_arr[i] = state[3]
        state = rk4_step(state, dt, m1, m2, L1, L2)

    x1 = L1 * np.sin(th1_arr)
    y1 = -L1 * np.cos(th1_arr)
    x2 = x1 + L2 * np.sin(th2_arr)
    y2 = y1 - L2 * np.cos(th2_arr)

    # total energy (should be conserved)
    KE = (0.5 * m1 * L1**2 * w1_arr**2
          + 0.5 * m2 * (L1**2 * w1_arr**2 + L2**2 * w2_arr**2
                        + 2 * L1 * L2 * w1_arr * w2_arr * np.cos(th1_arr - th2_arr)))
    PE = -(m1 + m2) * G * L1 * np.cos(th1_arr) - m2 * G * L2 * np.cos(th2_arr)
    energy = KE + PE

    return SimResult(t_arr, th1_arr, th2_arr, w1_arr, w2_arr, x1, y1, x2, y2, energy)


def estimate_lyapunov(th1_0: float, th2_0: float, epsilon: float = 1e-9,
                      T: float = 50, dt: float = 0.005) -> float:
    n_steps = int(T / dt)
    state = np.array([th1_0, 0.0, th2_0, 0.0])
    state_p = state.copy()
    state_p[0] += epsilon

    lyap_sum = 0.0
    count = 0
    renorm_interval = 100

    for i in range(n_steps):
        state = rk4_step(state, dt, 1, 1, 1, 1)
        state_p = rk4_step(state_p, dt, 1, 1, 1, 1)

        if (i + 1) % renorm_interval == 0:
            diff = state_p - state
            dist = np.linalg.norm(diff)
            if dist > 0:
                lyap_sum += np.log(dist / epsilon)
                count += 1
                state_p = state + epsilon * diff / dist

    return lyap_sum / (count * renorm_interval * dt) if count > 0 else 0.0

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trajectory(res: SimResult, title: str = "Double Pendulum"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold")

    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # ── Trace of second mass ──
    ax = axes[0, 0]
    n = len(res.x2)
    colors = plt.cm.plasma(np.linspace(0, 1, n))
    for i in range(1, n, 20):
        ax.plot(res.x2[i-1:i+1], res.y2[i-1:i+1], color=colors[i], linewidth=0.5, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title("Tip Trajectory", color="white")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")

    # ── Angles vs time ──
    ax = axes[0, 1]
    ax.plot(res.t, np.degrees(res.th1), "#4a90d9", linewidth=0.8, label="θ₁")
    ax.plot(res.t, np.degrees(res.th2), "#ff6b35", linewidth=0.8, label="θ₂")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Angle (°)", color="white")
    ax.set_title("Angles vs Time", color="white")
    ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")
    ax.grid(True, alpha=0.15, color="white")

    # ── Phase space (θ₂ vs ω₂) ──
    ax = axes[1, 0]
    ax.scatter(np.degrees(res.th2), res.w2, c=res.t, cmap="plasma", s=0.2, alpha=0.6)
    ax.set_xlabel("θ₂ (°)", color="white")
    ax.set_ylabel("ω₂ (rad/s)", color="white")
    ax.set_title("Phase Space (mass 2)", color="white")

    # ── Energy conservation ──
    ax = axes[1, 1]
    e_ref = res.energy[0]
    ax.plot(res.t, (res.energy - e_ref) / abs(e_ref) * 100, "#2ecc71", linewidth=1)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Energy Drift (%)", color="white")
    ax.set_title("Energy Conservation Check", color="white")
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    return fig


def plot_chaos_comparison(th_base: float = 2.0, deltas=(0.0, 0.001, 0.01, 0.1)):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(deltas)))

    for delta, col in zip(deltas, cmap):
        res = simulate(th_base + delta, th_base, T=20)
        label = f"Δθ = {delta:.4f} rad" if delta > 0 else "Reference"
        ax.plot(res.t, np.degrees(res.th2), color=col, linewidth=1.2, label=label)

    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("θ₂ (°)", color="white")
    ax.set_title("Sensitive Dependence on Initial Conditions", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, alpha=0.15, color="white")
    ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    return fig

# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DOUBLE PENDULUM CHAOS EXPLORER")
    print("=" * 60)

    # ── Standard run ──
    th1_0 = np.radians(120)
    th2_0 = np.radians(120)

    print(f"\n  Initial angles: θ₁ = 120°, θ₂ = 120°")
    print(f"  Simulation: 30 s at dt = 0.005 s (RK4)\n")

    res = simulate(th1_0, th2_0, T=30)

    print(f"  Energy drift: {abs(res.energy[-1] - res.energy[0]) / abs(res.energy[0]) * 100:.6f}%")
    print(f"  Max tip displacement: {max(np.sqrt(res.x2**2 + res.y2**2)):.3f} m")

    # ── Lyapunov exponent ──
    print(f"\n  Estimating largest Lyapunov exponent...")
    lyap = estimate_lyapunov(th1_0, th2_0, T=50)
    print(f"  λ ≈ {lyap:.3f} /s {'(chaotic)' if lyap > 0 else '(regular)'}")

    # ── Plots ──
    print("\nGenerating plots...")

    fig1 = plot_trajectory(res, "Double Pendulum — θ₁₀ = 120°, θ₂₀ = 120°")
    fig1.savefig("double_pendulum.png", dpi=150, bbox_inches="tight")
    print("  → double_pendulum.png")

    fig2 = plot_chaos_comparison(th1_0)
    fig2.savefig("chaos_sensitivity.png", dpi=150, bbox_inches="tight")
    print("  → chaos_sensitivity.png")

    # ── Low-energy (non-chaotic) comparison ──
    res_low = simulate(np.radians(15), np.radians(15), T=30)
    lyap_low = estimate_lyapunov(np.radians(15), np.radians(15), T=50)
    print(f"\n  Low-energy case (15°, 15°): λ ≈ {lyap_low:.3f} /s {'(chaotic)' if lyap_low > 0 else '(regular)'}")

    fig3 = plot_trajectory(res_low, "Double Pendulum — θ₁₀ = 15°, θ₂₀ = 15° (near-linear)")
    fig3.savefig("pendulum_linear.png", dpi=150, bbox_inches="tight")
    print("  → pendulum_linear.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
