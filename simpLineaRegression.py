import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# Parameters
n = 20  # Number of observations
beta_0 = 5  # Intercept
beta_1 = -2  # Slope
sigma_sq = 3  # Noise variance
fixed_x = np.random.exponential(scale=1, size=n)  # Fixed x values from an exponential distribution

# Function to simulate the linear regression model
def sim_lin_gauss(intercept=beta_0, slope=beta_1, noise_variance=sigma_sq, x=fixed_x, model=False):
    # Generate y values with Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_variance), size=len(x))
    y = intercept + slope * x + noise

    # Return either a fitted model or the data as a DataFrame
    if model:
        lm = LinearRegression()
        lm.fit(x.reshape(-1, 1), y)
        return lm
    else:
        return pd.DataFrame({"x": x, "y": y})

# Simulate slope estimates and predictions
n_simulations = 10_000
slope_samples = []
pred_samples = []

for _ in range(n_simulations):
    lm = sim_lin_gauss(model=True)
    slope_samples.append(lm.coef_[0])  # Extract the slope coefficient
    pred_samples.append(lm.predict(np.array([[-1]]))[0])  # Predict for x = -1

# Plot the distributions
plt.figure(figsize=(10, 6))

# Slope estimate distribution
plt.subplot(2, 1, 1)
plt.hist(slope_samples, bins=50, density=True, alpha=0.7, label="Simulated slopes")
x_vals = np.linspace(-3.5, -0.5, 100)
theoretical_std = np.sqrt(sigma_sq / (n * np.var(fixed_x)))
plt.plot(x_vals, norm.pdf(x_vals, loc=beta_1, scale=theoretical_std), 'b-', label="Theoretical density")
plt.xlabel(r'$\hat{\beta}_1$')
plt.ylabel('Density')
plt.legend()

# Prediction distribution
plt.subplot(2, 1, 2)
plt.hist(pred_samples, bins=50, density=True, alpha=0.7, label="Simulated predictions")
x_vals = np.linspace(4,
