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
#TODO tensor created from numpy array has the type after numpy array if type not specified in tensor() function

# Example of 0 dimension tensor
dim_0 = torch.tensor([1, 2, 3]).sum()
print('Example of 0 dimension tensor {} and how to collect value from it by tensor.item()  {}\n'.format(dim_0, dim_0.item()))

# Examples of changing from CPU to GPU
a = torch.FloatTensor([2, 3])
print(f'CPU tensor {a}')
ca = a.to('cuda')
print(f'GPU tensor {ca}') # some error occurs while changing to GPU due to that pytorch do not
# support GPU acceleration on MACOS
