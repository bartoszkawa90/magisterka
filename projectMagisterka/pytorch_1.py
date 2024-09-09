import torch
import numpy as np


a = torch.FloatTensor(3, 2)
print(f'Example pytorch tensor {a}')

a.zero_() # method to clear tensor values
print(f'Clear tensor {a}')

# tensor can be created with values passed to constructor, in form of lists or numpy arrays
list_a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print(list_a)

np_array = np.zeros(shape=(3, 2))
np_a = torch.tensor(np_array)
print(f'Example tensor {np_a} made by passing numpy array {np_array}')
np.float32
