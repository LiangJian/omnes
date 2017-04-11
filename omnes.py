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
from matplotlib.backends.backend_pdf import PdfPages

# print(omega(4*m_pi**2 + 100, 0))
# print(part2(1))
# print(integrate_delta(0))

#print(integrate.quad(lambda xx: np.arctan(xx)/xx**2, 1, np.inf))
#print(of.principle_integration_delta_ori(4*of.m_pi**2 + 100, 0))


#x = [0, 1, 2, 3, 4]
#y = x
#np.polynomial.chebyshev.chebfit(x, y, 5)


#x = np.arange(of.s_minimum + 100, of.s_maximum, 2000)
D = 0.9
off = 0.04
x = np.arange(2*of.m_pi/1000+off, D, (D-2*of.m_pi/1000-off)/800.)
#x = np.arange(off, D, (D-off)/200.)
print(x.shape)
y_0_real = x.copy()
y_1_real = x.copy()
y_2_real = x.copy()
y_s_1_real = x.copy()
y_s_test_real = x.copy()

y_0_imaginary = x.copy()
y_1_imaginary = x.copy()
y_2_imaginary = x.copy()
y_s_1_imaginary = x.copy()
y_s_test_imaginary = x.copy


output_file = open("out.txt", 'w')

for i in range(0, x.size):
    if i%(x.size//10) == 0:
        print("%d%%"%((i//(x.size//10))*10)+" done...")
    temp_0 = of.omega_function((x[i]*1000)**2, 0)
    temp_1 = of.omega_function((x[i]*1000)**2, 1)
    temp_2 = of.omega_function((x[i]*1000)**2, 2)
    temp_s = of.omega_1_Susan((x[i]*1000)**2)
    y_0_real[i] = np.real(temp_0)
    y_0_imaginary[i] = np.imag(temp_0)
    y_1_real[i] = np.real(temp_1)
    y_1_imaginary[i] = np.imag(temp_1)
    y_2_real[i] = np.real(temp_2)
    y_2_imaginary[i] = np.imag(temp_2)
    y_s_1_real[i] = np.real(temp_s)
    y_s_1_imaginary[i] = np.imag(temp_s)
    print(x[i], y_0_real[i], y_0_imaginary[i], y_1_real[i], y_1_imaginary[i], y_2_real[i], y_2_imaginary[i],
    file=output_file)




pdf = PdfPages('omnes.pdf')

plt.xlim(0,2.83)
plt.ylim(-90,270)
plt.xlabel(r"$\sqrt{s}$/GeV")
plt.ylabel(r"$\delta(s)/degree$")
s_test = np.arange(of.m_pi*2/1000+0.0001, 26.0, 0.001)
delta_test = s_test.copy()
for i in range(0, s_test.size):
    delta_test[i] = of.delta((s_test[i]*1000)**2, 0)*180/np.pi
plt.plot(s_test, delta_test, label="I=0")
for i in range(0, s_test.size):
    delta_test[i] = of.delta((s_test[i]*1000)**2, 1)*180/np.pi
plt.plot(s_test, delta_test, label="I=1")
for i in range(0, s_test.size):
    delta_test[i] = of.delta((s_test[i]*1000)**2, 2)*180/np.pi
plt.plot(s_test, delta_test, label="I=2")
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

plt.xlim(0.25,.9)
plt.ylim(0.,4.)
plt.xlabel(r"$\sqrt{S}$/GeV")
plt.ylabel(r"$\Omega_{real}(s)$")
plt.plot(x, y_0_real,label="I=0")
plt.plot(x, y_1_real,label="I=1")
plt.plot(x, y_2_real,label="I=2")
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

plt.xlim(0.25,0.9)
plt.ylim(-8,8.)
plt.xlabel(r"$\sqrt{s}$/GeV")
plt.ylabel(r"$\Omega_{imag}(s)$")
plt.plot(x, y_0_imaginary,label="I=0")
plt.plot(x, y_1_imaginary,label="I=1")
plt.plot(x, y_2_imaginary,label="I=2")
plt.legend()
plt.grid()
pdf.savefig()
plt.close()
#plt.show()

plt.xlim(0.25,0.9)
plt.ylim(0.,4.)
plt.xlabel(r"$\sqrt{s}$/GeV")
plt.ylabel(r"$\Omega1_{real}(s)$")
plt.plot(x, y_1_real,label="Colangelo")
plt.plot(x, y_s_1_real,label="Susan")
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

plt.xlim(0.25,0.9)
plt.ylim(-8.,8.)
plt.xlabel(r"$\sqrt{s}$/GeV")
plt.ylabel(r"$\Omega1_{imag}(s)$")
plt.plot(x, y_1_imaginary,label="Colangelo")
plt.plot(x, y_s_1_imaginary,label="Susan")
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

#plt.show()
pdf.close()

#plt.plot(x, y_0)
#plt.show()
print('hello')
#print(of.omega_function(of.s_maximum,0))

print(2*of.m_pi)
print(np.sqrt(of.s_maximum))
