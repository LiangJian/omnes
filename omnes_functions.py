#!//opt/local/bin/python3
import numpy as np
import scipy.integrate as integrate


m_pi_plus = 139.0
m_pi_zero = 135.0
m_pi = 1/3.*(2*m_pi_plus + m_pi_zero)

A = np.array([.220, .0379, -.0444])
B = np.array([.268, .140e-4, -.0857])
C = np.array([-.0139, -.673e-4, -.221e-2])
D = np.array([-.139e-2, .163e-7, -.129e-3])
S_zero = np.array([36.77, 30.72, -21.62])
S_zero *= m_pi
limit = np.inf


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
    return np.arctan(tan_delta(s, index))


def delta_s_0(s):
    return delta(s, 0)/s


def delta_s_1(s):
    return delta(s, 1)/s


def delta_s_2(s):
    return delta(s, 2)/s


def integrate_delta(index):
    if index == 0:
        return integrate.quad(delta_s_0, 4*m_pi**2, limit)[0]
    if index == 1:
        return integrate.quad(delta_s_1, 4*m_pi**2, limit)[0]
    if index == 2:
        return integrate.quad(delta_s_2, 4*m_pi**2, limit)[0]


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
            integral_2 = integrate.quad(tmp, s+epsilon, limit)[0]

        if index == 1:
            def tmp(s_prime_):
                return delta_s_prime_1(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, limit)[0]

        if index == 2:
            def tmp(s_prime_):
                return delta_s_prime_2(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, limit)[0]

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


def delta_s_prime_ori_0(s, s_prime):
    return delta(s_prime, 0)/(s_prime*(s_prime - s))


def delta_s_prime_ori_1(s, s_prime):
    return delta(s_prime, 1)/(s_prime*(s_prime - s))


def delta_s_prime_ori_2(s, s_prime):
    return delta(s_prime, 2)/(s_prime*(s_prime - s))


def principle_integration_delta_ori(s, index):

    epsilon = 0.01
    old = 0.0
    cut = 1e-8
    step = 0.5

    while True:
        integral_1 = .0
        integral_2 = .0

        if index == 0:
            def tmp(s_prime_):
                return delta_s_prime_ori_0(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, limit)[0]

        if index == 1:
            def tmp(s_prime_):
                return delta_s_prime_ori_1(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, limit)[0]

        if index == 2:
            def tmp(s_prime_):
                return delta_s_prime_ori_2(s, s_prime_)
            integral_1 = integrate.quad(tmp, 4*m_pi**2, s-epsilon)[0]
            integral_2 = integrate.quad(tmp, s+epsilon, limit)[0]

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