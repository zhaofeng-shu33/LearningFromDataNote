import numpy as np
a = np.array([1, 2, 3])
print(np.linalg.norm(a))

a = np.array([[1, 2], [3, 4]])
print(np.linalg.eig(a)[0])

a = np.array([[1, 2], [3, 4], [5, 6]])
print(np.sum(a, axis=1))

import scipy.stats
x = np.linspace(-3, 3)
y = scipy.stats.norm.pdf(x)
print(x, y)

import matplotlib.pyplot as plt
c = np.random.normal(size=1000)
plt.hist(c, density=True)
plt.plot(x, y)
plt.show()
