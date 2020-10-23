from lfdnn import tensor, operator
a = tensor([3, 4], 't')
print(a.shape)
import numpy as np
b = operator.relu(a)
feed = {'t': np.random.normal(size=[3, 4])}
print(b.eval(feed))
print(b.differentiate(a, feed))
print(a.back(b, feed))
w = tensor([4, 1], 'w')
b = tensor([1, 1], 'b')
h = operator.add(operator.matmul(a, w), b)
y = operator.sigmoid(h)
feed.update({'w': np.ones([4, 1]),
             'b': np.array([[2]])})
y.eval(feed)
