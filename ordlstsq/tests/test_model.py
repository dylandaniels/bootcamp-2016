from ordlstsq import model
import numpy as np

def test_answer():
    X = np.array([[ 1.   ,  0.389],
                 [ 1.   ,  0.2  ],
                 [ 1.   ,  0.241],
                 [ 1.   ,  0.463],
                 [ 1.   ,  4.585],
                 [ 1.   ,  1.097],
                 [ 1.   ,  1.642],
                 [ 1.   ,  4.972],
                 [ 1.   ,  7.957],
                 [ 1.   ,  5.585],
                 [ 1.   ,  5.527],
                 [ 1.   ,  6.964]])
    y = [11.416,
         4.514,
         12.204,
         14.835,
         8.416,
         6.563,
         17.343,
         13.02,
         15.19,
         11.902,
         22.721,
         22.324]
    print(model)
    beta = np.array([ 10.07128585,   0.99925723])
    np.testing.assert_allclose(model.fit(X, y), beta)

def test_mean():
    X = np.ones(100)
    y = np.random.rand(100)
    known_beta = np.array(np.mean(y))
    np.testing.assert_allclose(model.fit(X, y), known_beta)
