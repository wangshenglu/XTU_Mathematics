import numpy as np

def comp_tra(a, b, n, func):
    """
    [a, b] : 积分区间，float
    n : 区间等分数，int
    func : 被积函数，function
    Returns : 复化梯形公式计算积分值，float
    """
    x = np.linspace(a, b, n+1)[1:-1]
    h = (b - a) / n
    return h/2 * (func(a) + 2*sum(func(x)) + func(b))

def comp_simp(a, b, n, func):
    """
    [a, b] : 积分区间，float
    n : 区间等分数，int
    func : 被积函数，function
    Returns : 复化Simpson公式计算积分值，float
    """
    x = np.linspace(a, b, n+1)[1:-1]
    x_ = (np.linspace(a, b, n+1) + (b-a) / (2*n))[:-1]
    h = (b - a) / n
    return h/6 * (func(a) + 4*sum(func(x_)) \
                  + 2*sum(func(x)) + func(b))

def adap_tra(a, b, func, epsilon):
    """
    [a, b] : 积分区间，float
    func : 被积函数，function
    epsilon : 预置的容许误差
    Returns : 自适应Simpsion公式计算积分值，float
    """
    n = 1
    T1 = (b-a) / 2 * (func(a) + func(b))
    while True:
        T2 = T1 / 2 + (b-a)/n/2  * sum(func(np.linspace(a, b, 2*n+1)[1::2]))
        n *= 2
        if np.abs(T1-T2) < epsilon:
            return T2, n
        T1 = T2

def adap_simp(a, b, func, epsilon):
    """
    [a, b] : 积分区间，float
    func : 被积函数，function
    epsilon : 预置的容许误差
    Returns : 自适应Simpson公式计算积分值，float
    """
    n = 1
    S1 = (b-a) / 6 * (func(a) + 4*func(a/2+b/2) + func(b))
    while True:
        h = (b-a) / n
        x = np.linspace(a, b, n+1)[:-1]
        S = 2 * sum(func(x+h/4)) - sum(func(x+h/2)) + 2 * sum(func(x+3*h/4))
        S2 = S1 / 2 + (b-a)/n/6  * S
        n *= 2
        if np.abs(S1-S2) < epsilon:
            return S2, n
        S1 = S2

def Romberg(a, b, func, e=1e-9):
    """
    [a, b] : float, 积分区间
    func : function, 被积函数
    e : float, default=1e-7, 计算精度要求
    Returns : float, Romberg外推法计算积分值
    """
    T = [comp_tra(a, b, 2**k, func) for k in range(5)]
    S = [4/3 * T[i+1] - 1/3 * T[i] for i in range(4)]
    C = [16/15 * S[i+1] - 1/15 * S[i] for i in range(3)]
    R = [64/63 * C[i+1] - 1/63 * C[i] for i in range(2)]
    k = 5
    while np.abs(R[-1] - R[-2]) >= e:
        T.append(comp_tra(a, b, 2**k, func))
        S.append(4/3 * T[-1] - 1/3 * T[-2])
        C.append(16/15 * S[-1] - 1/15 * S[-2])
        R.append(64/63 * C[-1] - 1/63 * C[-2])
        k += 1
    return R[-1]

# Gauss-Legendre积分节点和系数表
parameter = [[[0], [2]],
            [[-0.5773502692, 0.5773502692], [1, 1]],
            [[-0.7745906692, 0, 0.7745906692], [5/9, 8/9, 5/9]],
            [[-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116], [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]],
            [[-0.9061798459, -0.5384693101, 0, 0.5384693101, 0.9061798459], [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]]]

def Gauss_Leg(a, b, func, n):
    """
    [a, b] : float, 积分区间
    func : function, 被积函数
    n : int, 数值点个数
    Returns : float, Gauss-Legendre积分值
    """
    func_t = lambda t : func((a+b)/2 + (b-a)/2 * t)
    xk, Ak = parameter[n-1]
    return (b-a)/2 * sum([Ak[i] * func_t(xk[i]) for i in range(n)])
