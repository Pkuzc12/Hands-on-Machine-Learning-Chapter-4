import numpy as np
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, ElasticNet

np.random.seed(42)

m = 100
X = 6*np.random.rand(m, 1)-3
y = 0.5*X**2+X+2+np.random.randn(m, 1)

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y.ravel())
print(ridge_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y.ravel())
print(lasso_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty="l1")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y.ravel())
print(elastic_net.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty="elasticnet")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))
