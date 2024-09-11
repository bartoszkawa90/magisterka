import torch
import torch.nn as nn
import numpy as np


# l = nn.Linear(2, 5)
# v = torch.FloatTensor([1, 2])
# print(f'Example of passing tensor v to layer l =  {l(v)}')

"""
Przypisując moduły podrzędne do poszczególnych pól rejestrujemu moduł
Po wykonaniu kodu konstruktora wszystkie pola zostaną automatycznie zarejestrowane
"""
class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    v = torch.tensor([[2, 3]], dtype=torch.float32)
    out = net(v)
    print('Example of our new module instance and what module returned')
    print(net)
    print(out)
