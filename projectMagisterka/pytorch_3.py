import torch
import torch.nn as nn


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