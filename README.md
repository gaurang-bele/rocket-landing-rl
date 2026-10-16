# Tesla-Style Autonomous Rocket Landing via Reinforcement Learning

An end-to-end Reinforcement Learning system designed to train an AI agent to land a rocket in a physics simulator (**Gymnasium LunarLanderContinuous-v3**) using the engineering paradigms Tesla uses to train autonomous robots (**Optimus**) and Full Self-Driving (FSD).

---

## Key Features

1. **Pure Emergent Flight Dynamics (Zero Hardcoded Rules)**:
   The neural network is never given flight rules or trajectory templates. It discovers aerodynamic stabilization, retro-propulsion, and the "suicide burn" (hoverslam) purely through trial-and-error exploration.
2. **Anti-Reward-Hacking Engine**:
   - **Hover Exploit Prevention**: Imposes continuous fuel consumption penalties proportional to main and side thruster throttles. Hovering indefinitely drains score rapidly.
   - **Kamikaze Suicide Prevention**: Calibrates terminal crash penalties ($-150$) and safe touchdown bonuses ($+150$) with strict invariants (vertical descent velocity, upright tilt angle, both legs grounded) ensuring fast crashes are strictly worse than any careful landing attempt.
   - **Spin-to-Win Prevention**: Potential-based kinetic energy shaping penalizes high angular velocities and unstable tilt.
3. **Sim-to-Real Domain Randomization**:
   - Dynamic turbulent crosswinds and wind gusts.
   - Actuator thrust variations ($\pm 15\%$ throttle dispersion).
   - Sensor noise (radar altimeter jitter, IMU gyroscope drift).
   - High-altitude randomized drop angles.
4. **Curriculum Learning**:
   Automatically ramps difficulty across 3 distinct stages:
   - **Stage 1 (0k - 100k steps)**: Calm, low-altitude descent to master engine modulation.
   - **Stage 2 (100k - 300k steps)**: Moderate crosswinds and lateral velocity drift.
   - **Stage 3 (300k - 600k+ steps)**: Full Tesla stress test with turbulent storms and hardware degradation.
5. **Real-time Avionics Telemetry HUD**:
   Pygame-powered cockpit visualizer rendering artificial horizon, pitch ladder, vertical speed indicators, throttle gauges, fuel consumption, and trajectory ribbon trails.

---

## Project Structure

```
rocket-landing-rl/
├── config/
│   └── settings.py          # Centralized hyperparams, physics bounds, and reward weights
├── envs/
│   ├── __init__.py          # Environment factory for SB3 vectorization
│   ├── rocket_wrapper.py    # Sim-to-Real domain randomization wrapper
│   └── reward_engine.py     # Potential-based shaping & anti-hacking reward engine
├── telemetry/
│   ├── __init__.py
│   └── avionics_hud.py      # Real-time Pygame avionics HUD dashboard
├── tests/
│   └── test_anti_hacking.py # Empirical unit tests verifying exploit prevention
├── train.py                 # Multi-worker PPO training with curriculum scheduler
├── evaluate.py              # 4-scenario Sim-to-Real stress benchmark
├── run.py                   # Master CLI interface
└── requirements.txt         # Dependencies
```

---

## Quickstart

### 1. Activate Environment
In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Verify Anti-Reward-Hacking Engine
Run the exploit prevention unit tests:
```powershell
python run.py --mode test
```
Outputs confirmation that infinite hovering, kamikaze suicide crashing, and spin-to-win exploits are mathematically eliminated.

### 3. Train the Model
Train the policy using 8 parallel CPU workers with PPO:
```powershell
python run.py --mode train --timesteps 600000 --envs 8
```
- Periodic evaluations automatically save the best performing model to `models/best_model.zip`.
- Checkpoints are saved every 50,000 steps.
- TensorBoard metrics logged to `logs/`.

To inspect training in real-time with TensorBoard:
```powershell
tensorboard --logdir logs
```

### 4. Watch Live Avionics Telemetry Demo
Watch your trained policy land the rocket in real time with the cockpit HUD:
```powershell
python run.py --mode demo --model models/best_model.zip --episodes 5
```

### 5. Run Stress Benchmark
Test the agent against 4 harsh Sim-to-Real scenarios (Nominal Calm, Turbulent Storm, 25° Emergency Tilt Recovery, Actuator Degradation):
```powershell
python run.py --mode benchmark --model models/best_model.zip --episodes 20
```
