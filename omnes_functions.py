#!/opt/local/bin/python3
import numpy as np
import scipy.integrate as integrate


m_pi_plus = 139.570
m_pi_zero = 134.977
m_pi = 1/3.*(2*m_pi_plus + m_pi_zero)
m_eta = 547.862

s_minimum = 4*m_pi**2
s_maximum = (m_eta - m_pi)**2

A = np.array([.220, .0379, -.0444])
B = np.array([.268, .140e-4, -.0857])
C = np.array([-.0139, -.673e-4, -.221e-2])
D = np.array([-.139e-2, .163e-7, -.129e-3])
S_zero = np.array([36.77, 30.72, -21.62])
S_zero *= m_pi**2

A /= m_pi**(0)
B /= m_pi**(2)
C /= m_pi**(4)
D /= m_pi**(6)
A[1] /= m_pi**(2)
B[1] /= m_pi**(2)
C[1] /= m_pi**(2)
D[1] /= m_pi**(2)

upper_limit = np.inf


def q(s):
    return np.sqrt(s/4. - m_pi**2)


def factor_head(s):
    return np.sqrt(1 - 4*m_pi**2/s)


def factor_body(s, index):
    global A, B, C, D
    q_ = q(s)
    tmp = A[index] + B[index]*q_**2 + C[index]*q_**4 + D[index]*q_**6
    if index == 1:
        tmp *= q_**2
    return tmp


def factor_tail(s, index):
    global S_zero
    return (4*m_pi**2 - S_zero[index])/(s - S_zero[index])


def tan_delta(s, index):
    return factor_head(s) * factor_body(s, index) * factor_tail(s, index)


def part3(s, index):
    tan_ = tan_delta(s, index)
    return (1 + 1j*tan_)/np.sqrt(1 + tan_**2)


def delta(s, index):
    temp_delta = np.arctan(tan_delta(s, index))
<<<<<<< Updated upstream
    if index == 0:
        if s > S_zero[0]:
            temp_delta += np.pi
    if index == 1:
        if s > S_zero[1]:
=======
    if index == 1:
        if temp_delta < 0:
>>>>>>> Stashed changes
            temp_delta += np.pi
    return temp_delta

'''''
def delta_s_0(s):
    return delta(s, 0)/s


def delta_s_1(s):
    return delta(s, 1)/s


def delta_s_2(s):
    return delta(s, 2)/s


def integrate_delta(index):
    if index == 0:
        return integrate.quad(delta_s_0, 4*m_pi**2, upper_limit)[0]
    if index == 1:
        return integrate.quad(delta_s_1, 4*m_pi**2, upper_limit)[0]
    if index == 2:
        return integrate.quad(delta_s_2, 4*m_pi**2, upper_limit)[0]


def part2(index):
    return np.exp(-1./np.pi*integrate_delta(index))


def delta_s_prime_0(s, s_prime):
    return delta(s_prime, 0)/(s_prime - s)


def delta_s_prime_1(s, s_prime):
    return delta(s_prime, 1)/(s_prime - s)


def delta_s_prime_2(s, s_prime):
    return delta(s_prime, 2)/(s_prime - s)


def principle_integration_delta(s, index):

    epsilon = 0.01
    old = 0.0
    cut = 1e-8
    step = 0.5

    while True:
        integral_1 = .0
        integral_2 = .0

        if index == 0:
            def tmp(s_prime_):
                return delta_s_prime_0(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, upper_limit)[0]

        if index == 1:
            def tmp(s_prime_):
                return delta_s_prime_1(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, upper_limit)[0]

        if index == 2:
            def tmp(s_prime_):
                return delta_s_prime_2(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, upper_limit)[0]

        new = float(integral_1 + integral_2)
        if new == float('NaN'):
            print("principle integration meets nan.")
            break
        if new == float('Inf') or new == -float('Inf'):
            print("principle integration meets inf.")
            break
        if np.abs(new - old) < cut:
            break
        else:
            old = new
            epsilon *= step

    return old


def part1(s, index):
    return np.exp(1./np.pi*principle_integration_delta(s, index))


def omega(s, index):
    return part1(s, index) * part2(index) * part3(s, index)
'''''


def delta_s_prime_ori_0(s, s_prime):
    return delta(s_prime, 0)/(s_prime*(s_prime - s))


def delta_s_prime_ori_1(s, s_prime):
    return delta(s_prime, 1)/(s_prime*(s_prime - s))


def delta_s_prime_ori_2(s, s_prime):
    return delta(s_prime, 2)/(s_prime*(s_prime - s))


def integration_to_inf(func, lower_limit):
    upper_limit_try = 1e10
    value = integrate.quad(func, lower_limit, upper_limit_try)[0]
    while(True):
        upper_limit_try *= 10
        new_value = integrate.quad(func, lower_limit, upper_limit_try)[0]
        if abs(new_value - value) < 1e-8:
            return new_value
        else:
            value = new_value



def principle_integration_delta_ori(s, index):

    epsilon = 0.01
    old = 0.0
    cut = 1e-10
    step = 0.4

    while True:
        integral_1 = .0
        integral_2 = .0

        if index == 0:
            def tmp(s_prime_):
                return delta_s_prime_ori_0(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            #integral_2 = integrate.quad(tmp, s+epsilon, upper_limit)[0]
            integral_2 = integration_to_inf(tmp, s+epsilon)


        if index == 1:
            def tmp(s_prime_):
                return delta_s_prime_ori_1(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, upper_limit)[0]

        if index == 2:
            def tmp(s_prime_):
                return delta_s_prime_ori_2(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, upper_limit)[0]

        new = float(integral_1 + integral_2)
        if new == float('NaN'):
            print("principle integration meets nan.")
            break
        if new == float('Inf') or new == -float('Inf'):
            print(index)
            print("principle integration meets inf.")
            break
        if np.abs(new - old) < cut:
            break
        else:
            old = new
            epsilon *= step

    return old


def omega_function(s, index):
    return np.exp(s/np.pi * principle_integration_delta_ori(s, index)) * part3(s, index)

a_s = -0.672
b_s = -1.008
c_s = 30.45
s_tilde = -125.7

a_s /= 100.0*m_pi**2
c_s *= m_pi**2
s_tilde *= m_pi**2

def g_susan(s):
    u = factor_head(s)
    return -1.0/np.pi * u*np.log((1+u)/(1-u)) + 1j*u

def omega_1_Susan(s):
    return (c_s - 2.0*m_pi**2/np.pi)/s_tilde * (s_tilde-s)/(a_s*s**2+b_s*s+c_s-(s-4.0*m_pi**2)/4.0*g_susan(s))
