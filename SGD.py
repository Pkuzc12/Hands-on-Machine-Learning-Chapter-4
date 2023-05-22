import numpy as np
from sklearn.linear_model import SGDRegressor

np.random.seed(42)

X = np.random.rand(100, 1)
y = 4+3*X+np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 50
t0, t1 = 5, 50
m = 100


def learning_schedule(t):
    return t0/(t+t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta-eta*gradients

print(theta)

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)
