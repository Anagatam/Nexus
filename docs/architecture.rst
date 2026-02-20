.. _architecture:

Project Architecture & Solver Routing
=====================================

The Nexus Facade Pattern
------------------------

To achieve true institutional-grade efficiency, Nexus was engineered around the **Facade Pattern**. Instead of forcing users to independently ingest arrays, run disjointed matrix multiplications, and manually query thermodynamic metrics, everything routes through the ``NexusAnalyzer``.

This strictly conforms to the ``scikit-learn`` API design philosophy:

1. **.calibrate(returns)**: Ingests your return matrices and normalizes them into secure internal states.
2. **.compute(alpha)**: Maps massive mathematical vectors simultaneously. 
3. **.fetch(metric)**: Retrieves isolated computations on demand.

Dynamic Solver Routing
----------------------

Nexus computes highly complex non-linear penalty functions, specifically:

- **Entropic Value at Risk (EVaR)**
- **Entropic Drawdown at Risk (EDaR)**
- **Relativistic VaR (RLVaR)**

These computations require mapping data onto absolute mathematical cones (Exponential Cones natively, 3D Power Cones relativistically). 

**The Scaling Engine:**
When Nexus detects the ``cvxpy`` library alongside an active commercial optimization license (such as **MOSEK** or **GUROBI**), it natively routes these thermodynamic equations through absolute optimal solvers, guaranteeing global minima in milliseconds.

**The Open-Source Fallback:**
If institutional licenses are absent, Nexus seamlessly gracefully falls back to the open-source ``scipy.optimize.minimize`` SLSQP solver, achieving high-fidelity optimization without catastrophic failures.
