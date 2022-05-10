import numpy as np

def lagrange(x, y):
    # Lagrange插值
    M = len(x)
    p = np.poly1d(0.0)
    for j in range(M):
        pt = np.poly1d(y[j])
        for k in range(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= np.poly1d([1.0, -x[k]])/fac
        p += pt
    return p

def diff_q(xi, yi):
    """
    xi : list-like, 节点
    yi : list-like, 节点函数值
    Returns : float, 差商
    """
    r, n = 0, len(xi)
    for i in range(n):
        w = 1
        for j in range(n):
            if j == i:
                continue
            w *= xi[i] - xi[j]
        r += yi[i] / w
    return r

def Newton(xi, yi):
    """
    xi : list-like, 节点
    yi : list-like, 节点函数值
    Returns : function, Newton差值函数
    """
    n = len(xi)
    def new(x):
        """
        x : float, 自变量
        Returns : float, x点处的Newton差值函数值
        """
        y = 0
        for i in range(n):
            w = 1
            for j in range(i):
                w *= x - xi[j]
            y += diff_q(xi[:i+1], yi[:i+1]) * w
        return y
    return new

def pi(xi, k, x):
    #计算连乘
    a, b = 1, 1 #分子，分母
    n = len(xi)
    for j in range(n):
        if j != k:
            a *= x - xi[j]
            b *= xi[k] - xi[j]
    return a/b

def Hermite(xi, yi, dyi):
    """
    xi : 数组，插值节点
    yi : 数组，节点函数值
    dyi : 数组，节点导函数值
    Return : 一个函数，Hermite插值函数
    --------
    由于插值函数和x的取值无关，我们可以利用xi, yi, dyi生成一个python中的函数
    这个函数就是Hermite插值函数，输入x输出H(x)
    如此后续计算各点Hermite函数值，只需输入x
    而不需再次用xi, yi, dyi生成Hermite插值函数
    """
    def he(x):
        """
        x : 浮点数
        Return : 在点x处的Hermite函数值
        """
        Rs = 0
        n = len(xi)
        for k in range(n):
            alpha = 0
            for j in range(n):
                if j != k:
                    alpha += 1 / (xi[k]-xi[j])
            alpha = 1 - 2 * (x - xi[k]) * alpha
            alpha *= pi(xi, k, x) ** 2
            beta = (x - xi[k]) * pi(xi, k, x) ** 2
            Rs += yi[k] * alpha + dyi[k] * beta
        return Rs
    return he

def sph(xi, yi, dyi, x):
    """
    xi : 数组，插值节点
    yi : 数组，节点函数值
    dyi : 数组，节点导数值
    Return : 浮点数，分段Hermite插值函数值
    """
    for i in range(len(xi)): # 找到x所在区间ei
        if x <= xi[i]:
            break
    ei = np.array([xi[i-1], xi[i]])
    yei = np.array([yi[i-1], yi[i]])
    dyei = np.array([dyi[i-1], dyi[i]])
    return Hermite(ei, yei, dyei)(x) # 在区间ei上插值

def sp1(xi, yi, x):
    """
    xi : 数组，插值节点
    yi : 数组，节点函数值
    Return : 浮点数，分段一次插值函数值
    """
    for i in range(len(xi)): # 找到x所在区间ei
        if x <= xi[i]:
            break
    ei = np.array([xi[i-1], xi[i]])
    yei = np.array([yi[i-1], yi[i]])
    return lagrange(ei, yei)(x) # 在区间ei上插值

def sp2(xi, yi, x):
    """
    xi : list-like, 插值节点
    yi : list-like, 节点函数值
    x : float, 自变量
    Return : float, 分段二次插值函数值
    """
    for i in range(0, len(xi), 2): # 找到x所在区间ei
        if x < xi[i]:
            break
    ei = np.array([xi[i-2], xi[i-1], xi[i]])
    yei = np.array([yi[i-2], yi[i-1], yi[i]])
    return Newton(ei, yei)(x) # 在区间ei上三点二次Newton插值

def diff_vector(xi, yi, dy):
    """
    xi : list-like, 差值节点
    yi : list-like，节点函数值
    dy : 长度为2的list-like, x0和xn处的导数值（Ⅰ型边界条件）
    Retruns : list-like, 三弯矩方程右端常数项
    """
    n = len(xi)
    d = [diff_q(xi[i-1:i+2], yi[i-1:i+2]) for i in range(1, n-1)]
    d.insert(0, (yi[1]-yi[0])/(xi[1]-xi[0]) - dy[0])
    d.append(dy[-1] - (yi[n-1]-yi[n-2])/(xi[n-1]-xi[n-2]))
    return 6 * np.array(d)

def init_M(hi, d):
    """
    hi : list-like, 差值区间长度，hi[i] = xi[i+1] - xi[i]
    d : list-like，三弯矩方程右端常数项
    Retruns : list-like, 三弯矩方程的解
    """
    n = len(hi) + 1
    A = 2 * np.eye(n, n) + np.eye(n, n, 1) + np.eye(n, n, -1)
    for i in range(1,n-1):
        A[i, i+1] = hi[i] / (hi[i] + hi[i-1])
        A[i, i-1] = 1 - A[i, i+1]
    M = np.linalg.solve(A, d)
    return M

def cubic_spline(xi, yi, dy, x):
    """
    xi : list-like, 差值节点
    yi : list-like，节点函数值
    dy : 长度为2的list-like, x0和xn处的导数值（Ⅰ型边界条件）
    x : float, 自变量
    Retruns : float, 三次样条差值函数值
    """
    dq = diff_vector(xi, yi, dy)
    hi = np.diff(xi)
    M = init_M(hi, dq)
    n = len(xi)
    for i in range(n):
        if xi[i-1] < x and x < xi[i]:
            rs = ((xi[i]-x)**3 * M[i-1] + (x-xi[i-1])**3 * M[i]) / (6*hi[i-1])
            rs += ((xi[i]-x) * yi[i-1] + (x-xi[i-1])*yi[i]) / hi[i-1]
            rs -= hi[i-1]/6 * ((xi[i]-x) * M[i-1]+(x-xi[i-1])*M[i])
            return rs

