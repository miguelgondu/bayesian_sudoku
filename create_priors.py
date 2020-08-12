"""
This script creates the 9x9 prior
in log space.
"""

import numpy as np
import matplotlib.pyplot as plt

# Core idea: to model time vs. hints as exponentially decreasing
# which translates to linearity (with negative slope) in logspace.
def prior_in_log_space(h):
    return np.log(600 + ((600 - 3) / (17 - 80)) * (h - 17))


# Computing it.
domain = np.arange(17, 81)
prior = prior_in_log_space(domain)

_, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(domain, np.exp(prior))
ax2.plot(domain, prior)
plt.show()
print(domain)
print(prior)

# Saving it.
array = np.vstack((domain, prior)).T
print(array)

np.savetxt(
    "./data/priors/9x9.csv",
    array,
    fmt="%2d,%.5f"
)
