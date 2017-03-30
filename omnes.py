#!/opt/local/bin/python3

'''''''''''''''
  March 2017
   Jun Shi

'''''''''''''''

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import omnes_functions as of

# print(omega(4*m_pi**2 + 100, 0))
# print(part2(1))
# print(integrate_delta(0))

print(integrate.quad(lambda xx: np.arctan(xx)/xx**2, 1, np.inf))
print(of.principle_integration_delta_ori(4*of.m_pi**2 + 100, 0))


x = [0, 1, 2, 3, 4]
y = x
np.polynomial.chebyshev.chebfit(x, y, 5)


x = np.arange(4*of.m_pi**2 + 1, 4*of.m_pi**2 + 500, 1)
y = x.copy()

for i in range(0, x.size):
    y[i] = of.principle_integration_delta_ori(x[i], 0)

plt.plot(x, y)
plt.show()
