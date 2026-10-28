# Double Pendulum Chaos Explorer

Simulates a double pendulum from Lagrangian mechanics, visualises the chaotic trajectory, phase space and energy conservation, and estimates the largest Lyapunov exponent to quantify chaos.

![Double pendulum trajectory](double_pendulum.png)
![Chaos sensitivity](chaos_sensitivity.png)

## What it does

| Feature | Detail |
|---|---|
| **Lagrangian derivation** | Full equations of motion for two coupled masses on rigid rods, no small-angle approximation |
| **RK4 integration** | Fourth-order Runge-Kutta with energy conservation as a built-in accuracy check |
| **Chaos detection** | Largest Lyapunov exponent estimated by tracking divergence of nearby trajectories with periodic renormalisation |
| **Sensitivity plot** | Same initial condition perturbed by 0.001, 0.01 and 0.1 rad — shows trajectories diverge exponentially |
| **Phase space** | θ–ω scatter plot coloured by time, showing the attractor structure |
| **Energy check** | Verifies total mechanical energy is conserved to within numerical precision |
| **High vs low energy** | Compare chaotic (120°) and near-linear (15°) regimes |

## Example output

```
  Initial angles: θ₁ = 120°, θ₂ = 120°
  Simulation: 30 s at dt = 0.005 s (RK4)

  Energy drift: 0.000012%
  Max tip displacement: 1.998 m

  Estimating largest Lyapunov exponent...
  λ ≈ 3.142 /s (chaotic)

  Low-energy case (15°, 15°): λ ≈ 0.003 /s (regular)
```

## Usage

```bash
pip install -r requirements.txt
python pendulum.py
```

## Built with

- Python 3.10+
- NumPy
- Matplotlib

## What I learned

How to derive equations of motion from a Lagrangian rather than force balance — much cleaner for coupled systems. Why RK4 matters for long-time integration (Euler drifts badly). What chaos actually means in a dynamical system — positive Lyapunov exponent, not just "looks random". The double pendulum transitions from regular to chaotic as initial energy increases, and even tiny perturbations (0.001 rad ≈ 0.06°) lead to completely different trajectories after a few seconds.
