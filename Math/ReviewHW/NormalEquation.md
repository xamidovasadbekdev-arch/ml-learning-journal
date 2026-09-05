# Hydrodynamic Resistance Analysis via Linear Regression

This module presents an in-depth numerical and theoretical study of Ordinary Least Squares (OLS) Linear Regression applied to hydrodynamic resistance modeling (using the Froude number). It explores implementation correctness, closed-form algebraic equivalences, numerical stability, feature scaling, and feature-space transformations.

---

## Technical Overview

1. **Closed-Form Equivalence (Bonus Analysis):**
   * Verified that the univariate normal equation reduces algebraically to standard statistical covariance and variance formulations:
     $$\theta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad \theta_0 = \bar{y} - \theta_1 \bar{x}$$
   * Demonstrated that evaluating OLS via matrix operations $X = [\mathbf{1}, x]$ produces identical parameter estimates ($100\%$ precision match) to the standard univariate formulation.

2. **Numerical Optimization & Solver Benchmarks:**
   * Benchmark tests (`%timeit`) comparing linear algebra solvers for $\theta = (X^T X)^{-1} X^T y$:
     * `np.linalg.lstsq` (QR/SVD Decomposition): **~77 µs** (Fastest & most robust)
     * `np.linalg.solve` (LU Decomposition): **~145 µs**
     * `np.linalg.pinv` (Moore-Penrose SVD): **~160 µs**
     * `np.linalg.inv` (Explicit Matrix Inversion): **~164 µs** (Least stable & computationally expensive)

3. **Conditioning & Feature Scaling:**
   * **Unscaled Data:** Matrix $X^T X$ exhibited a Condition Number ($\text{cond}$) of **~633,070.88**, rendering it severely *ill-conditioned* due to scale disparities between the intercept vector and the raw feature range.
   * **Z-Score Standardization:** Applying $x_{\text{scaled}} = \frac{x - \mu}{\sigma}$ dropped the condition number to **$1.00$** (a **>630,000× improvement** in numerical stability).
   * **Cost Surface Topography:** Feature scaling transformed the Mean Squared Error (MSE) loss surface from an elongated elliptical valley into symmetric, circular contours, enabling optimal gradient paths.

4. **Linearity in Parameters vs. Features:**
   * Tested polynomial expansion ($x, x^2, x^3$) alongside log-transformed target variables ($\log(y)$).
   * Demonstrated that models using $x^k$ remain strictly *linear in parameters* ($\theta$) despite non-linear feature inputs.
   * Evaluated $R^2$ performance across model configurations and analyzed physical interpretability for engineering domain contexts.
