import numpy as np

x_i = np.array([0.038, 0.194, 0.425, 0.626,	1.253, 2.500, 3.740])
y_i = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])

eps = 0.00001

b1 = 0.9
b2 = 0.2

last = None
beta = np.array([b1, b2])

dr_db1 = lambda x: - x/(beta[1] + x)
dr_db2 = lambda x: beta[0]*x/(beta[1] + x)**2

ms = lambda: np.sqrt((last[0] - beta[0])**2 + (last[1] - beta[1])**2)

rate = lambda x: (beta[0] * x) / (beta[1] + x)
grad_r = lambda x_i: np.array([[dr_db1(x) for x in x_i], 
                               [dr_db2(x) for x in x_i]]).T

hess = lambda x: np.array([[0, x/(beta[1] + x)**2], 
                           [x/(beta[1] + x)**2, -2*beta[0]*x/(beta[1] + x)**3]])

while True:
    J = grad_r(x_i)
    delta = np.linalg.inv(J.T @ J) @ J.T
    r = np.array([y_i[i] - rate(x_i[i]) for i in range(len(x_i))])
    last = beta
    beta = beta - delta @ r
    print(beta)

    if ms() <= eps:
        print("final error: " + str(ms()))
        print("final value: " + str(beta))
        exit()