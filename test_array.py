import numpy as np

_input = np.array([[1,2,],[3,4], [5,9]])
old_min = np.min(_input, axis=0)
old_max = np.max(_input, axis=0)
new_min = -np.ones(2)
new_max = np.ones(2)




_in = _input
_in = np.divide(_input-old_min,old_max-old_min)
_in = np.multiply(_in,new_max-new_min) + new_min
print(_in)