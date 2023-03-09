import numpy as np
from matplotlib import pyplot as plt

def integrate(mes:np.ndarray, dt:np.ndarray, C:np.float64 = 0) -> np.ndarray:
    v = np.zeros_like(dt) + C
    v[1:] = np.cumsum((mes[1:] + mes[:-1])/2 *(dt[1:] - dt[:-1])) + C
    return v

noise = np.array(np.random.randn(10)) * 0.16
real = np.array([i for i in range(10)])
dt = np.linspace(0, 1, 10)
measure = real + noise

v = integrate(measure, dt, 0)
s = integrate(v, dt, 5)

print("measured acceleration: \n", measure)
print("1-integrated: \n", v)
print("2-integrated: \n", s)

plt.plot(dt, measure)
plt.plot(dt, v)
plt.plot(dt, s)
plt.show()


