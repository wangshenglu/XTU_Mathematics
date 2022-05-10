import matplotlib.pyplot as plt
import numpy as np
from Interpolate import *

func = lambda x : 1/(1+x**2)
xi = np.linspace(-5, 5, 5)
x = np.linspace(-5, 5, 101)
plt.plot(x,[cubic_spline(xi,func(xi),[1,-1], _)for _ in x])
plt.show()