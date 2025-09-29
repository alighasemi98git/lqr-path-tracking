# LQR Path Tracking – Model Predictive Control

This project is part of the **EL2700 – Model Predictive Control** course at KTH. The focus is on applying the **Linear Quadratic Regulator (LQR)** to track an optimal raceline with both feedforward and feedback control. The assignment demonstrates how LQR can stabilize vehicle dynamics in the presence of disturbances while ensuring smooth and efficient path following.

## Project Overview

* **Problem Type:** Optimal control (LQR)
* **Context:** Vehicle dynamics & path tracking
* **Key Features:**

  * Linearized dynamic bicycle model
  * Feedforward optimization for raceline tracking
  * Feedback LQR controller for disturbance rejection
  * Constraint handling (steering angle limits)
  * Tuning via Bryson’s rule

## Tasks

1. **System Modeling**

   * Define continuous-time system matrices
   * Convert to discrete-time using `c2d`
   * States: displacement error, heading error, lateral velocity, yaw rate, steering angle
   * Inputs: steering rate, differential torque

2. **Feedforward Control**

   * Solve a quadratic program with CVXPY
   * Minimize output error + control effort
   * Generate optimal state and control trajectories `(xff, uff)`

3. **LQR Feedback Control**

   * Design state-feedback controller to stabilize error dynamics
   * Test with different cost matrices (Q, R) to evaluate bandwidth and tracking performance
   * Compare overshoot, settling time, and input usage

4. **Bryson’s Rule Tuning**

   * Normalize states and inputs using their maximum values
   * Tune controller by adjusting relative penalties on states vs. inputs
   * Evaluate trade-offs:

     * Large `li` (input penalty) → smoother inputs, worse tracking
     * Large `pi` (state penalty) → better tracking, more aggressive control
   * Find balanced tuning that avoids steering saturation

## Results

* **Feedforward trajectories** provide smooth baseline tracking of the raceline.
* **LQR feedback** successfully rejects disturbances and corrects deviations from the raceline.
* Controller performance depends heavily on tuning — striking a balance between steering effort and tracking accuracy is key.

## Repository Structure

```
.
├── src/
│   ├── task3.py              # Entry point script
│   ├── linear_car_model.py   # Vehicle model & system matrices
│   ├── sim.py                # Feedforward optimization & simulation
│   ├── qp_solver.py          # Raceline optimizer (provided)
│   ├── road.py               # Racetrack object (provided)
│
├── report/
│   └── Assignment3_GroupX.pdf
│
├── results/
│   ├── feedforward_trajectories.png
│   ├── lqr_tracking.png
│   ├── tuning_analysis.txt
│
└── README.md
```

## Usage Instructions

### Requirements

* Python 3.10+
* [CVXPY](https://www.cvxpy.org/)
* [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

1. Run the main script:

   ```bash
   python task3.py
   ```

2. The script will:

   * Compute the feedforward trajectory
   * Simulate LQR feedback tracking
   * Plot system states, inputs, and errors
   * Save results to the `results/` directory

## Technologies

* **Python 3**
* **CVXPY** for optimization
* **LQR control theory** (discrete-time Riccati equation)
* **Simulation framework** for vehicle path tracking

## Authors

* Group X (EL2700 – Model Predictive Control, 2025)

---

📄 This repository contains the Python implementation, report, and results for Assignment 3 of the MPC course, focusing on LQR-based path tracking and disturbance rejection.
