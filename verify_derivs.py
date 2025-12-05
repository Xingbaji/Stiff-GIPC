import numpy as np

def cubic_energy(g, ghat):
    y = g - ghat
    if y < 0:
        return -2.0 * (y**3) / (3.0 * ghat)
    return 0.0

def cubic_gradient(g, ghat):
    y = g - ghat
    if y < 0:
        return -2.0 * y**2 / ghat
    return 0.0

def cubic_curvature(g, ghat):
    y = g - ghat
    if y < 0:
        return 4.0 * (1.0 - g / ghat)
    return 0.0

# Parameters
dHat_squared = 1e-4  # dHat variable in code
dHat = np.sqrt(dHat_squared) # The actual distance threshold (ghat)
dis = 0.5 * dHat     # Current distance g
Kappa = 100.0

# Target Derivatives
# We want derivatives with respect to I5? Or y = dis / dHat_sqrt?
# In _calBarrierHessian:
#   tmp.v[8] = dis / dHat_sqrt  (which is y = dis/dHat)
#   It constructs H with lambda0 acting on this component.
#   So lambda0 should be d^2 E / dy^2.

# Verify E(y)
# E(g) = cubic_energy(g, dHat)
# g = y * dHat
# E(y) = cubic_energy(y * dHat, dHat)

def E_of_y(y):
    g = y * dHat
    return Kappa * cubic_energy(g, dHat)

# Numerical 2nd Derivative
y_val = dis / dHat
epsilon = 1e-6
d2Edy2_num = (E_of_y(y_val + epsilon) - 2*E_of_y(y_val) + E_of_y(y_val - epsilon)) / (epsilon**2)

# Analytical Expectation
# dE/dy = dE/dg * dg/dy = cubic_gradient(g, dHat) * dHat
# d2E/dy2 = cubic_curvature(g, dHat) * dHat * dHat
d2Edy2_ana = Kappa * cubic_curvature(dis, dHat) * dHat * dHat

# Implemented Formula
# return Kappa * curv * dHat / (dis * dis) * 0.25
# Note: input dHat in code is dHat_squared
impl_val = Kappa * cubic_curvature(dis, dHat) * dHat_squared / (dis**2) * 0.25

print(f"Numerical d2E/dy2: {d2Edy2_num}")
print(f"Analytical d2E/dy2: {d2Edy2_ana}")
print(f"Implemented Val:   {impl_val}")
print(f"Ratio Ana/Impl:    {d2Edy2_ana / impl_val}")

# Gradient Check
# We want coeff such that coeff * y = dE/dy (based on comment "coeff * (dis/dHat_sqrt) as pk1")
# pk1 usually implies force magnitude, but let's check the comment's claim.
# If coeff * y = dE/dy
# Then coeff = (dE/dy) / y = (grad * dHat) / y = grad * dHat / (g/dHat) = grad * dHat^2 / g

# Implemented Gradient Coeff
# return Kappa * grad * dHat_sqrt / dis
# Input dHat is dHat_squared. dHat_sqrt is dHat.
impl_grad = Kappa * cubic_gradient(dis, dHat) * dHat / dis

target_grad_coeff = Kappa * cubic_gradient(dis, dHat) * dHat**2 / dis

print(f"\nGradient Coeff Check:")
print(f"Target (dE/dy / y): {target_grad_coeff}")
print(f"Implemented:        {impl_grad}")
print(f"Ratio Target/Impl:  {target_grad_coeff / impl_grad}")

