import numpy as np
a = np.array([1, 2, 3])
print(np.linalg.norm(a))

A = np.array([[1, 2], [3, 4]])
print(np.linalg.eig(A)[0])

A = np.array([[1, 2], [3, 4], [5, 6]])
print(np.sum(A, axis=1))

import scipy.stats
x = np.linspace(-3, 3)
y = scipy.stats.norm.pdf(x)
print(x, y)

import matplotlib.pyplot as plt
c = np.random.normal(size=1000)
plt.hist(c, density=True)
plt.plot(x, y)
plt.show()
