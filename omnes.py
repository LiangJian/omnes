#!/opt/local/bin/python3

'''''''''''''''
  March 2017
   Jian Liang
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

'''''''''
x = np.arange(of.s_minimum + 1, of.s_maximum - 1, 100)
y_0_real = x.copy()
y_1_real = x.copy()
y_2_real = x.copy()

y_0_imaginary = x.copy()
y_1_imaginary = x.copy()
y_2_imaginary = x.copy()

for i in range(0, x.size):
    temp_0 = of.omega_function(x[i], 0)
    temp_1 = of.omega_function(x[i], 1)
    temp_2 = of.omega_function(x[i], 2)
    y_0_real[i] = np.real(temp_0)
    y_0_imaginary[i] = np.imag(temp_0)
    y_1_real[i] = np.real(temp_1)
    y_1_imaginary[i] = np.imag(temp_1)
    y_2_real[i] = np.real(temp_2)
    y_2_imaginary[i] = np.imag(temp_2)
'''''

s_test = np.arange(of.s_minimum+10, 800000, 100)
delta_test = s_test.copy()
for i in range(0, s_test.size):
    delta_test[i] = of.delta(s_test[i], 1)


plt.plot(s_test, delta_test)
plt.show()


#plt.plot(x, y_0)
#plt.show()
print('hello')
# print(of.omega_function(of.s_maximum-100,0))
