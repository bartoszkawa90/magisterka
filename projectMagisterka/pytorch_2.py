import torch


v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])
print(f'Example of tensors one with requires_grad {v1} and second one without {v2}')

v_sum = v1 + v2
v_res = (v_sum*2).sum()
print(f'Sum of vectors {v_sum} and value of v_res {v_res}')

# check which tensors are made bu user and which ones needs gradients
for t, name in zip([v1, v2, v_sum, v_res], ['v1', 'v2', 'v_sum', 'v_res']):
    print(f'Values of is_leaf {t.is_leaf} and requires_grad {t.requires_grad} for {name} tensor')

# how to count gradients for graph
v_res.backward()
print(f'Values of gradients counted for created graph  {v1.grad}')
