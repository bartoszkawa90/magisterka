import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteActor(nn.Module):
    def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
        super(DiscreteActor, self).__init__()
        self.device = device

        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))  # Spłaszcza do 4x4 niezależnie od wejścia
        )

        # Gałąź przetwarzająca prędkość – oczekujemy tensor o wymiarze [B, 1]
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        # Łączymy wyjście z CNN (flattenowane do 256*4*4) z przetworzoną prędkością (32)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )

    def forward(self, x, speed=None, manouver=None):
        # Normalizacja obrazu oraz przetwarzanie przez CNN
        x = x.to(self.device, dtype=torch.float32) / 255.0
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(x.size(0), -1)  # Flatten

        # Obsługa speed: jeśli brak, ustawiamy tensor zerowy
        if speed is None:
            speed = torch.zeros((x.size(0), 1), device=self.device)
        else:
            speed = speed.to(self.device, dtype=torch.float32)
            if speed.dim() == 1:
                speed = speed.unsqueeze(1)  # Upewnij się, że kształt to [B, 1]

        speed_features = self.speed_fc(speed)

        # Łączenie cech obrazu z cechami prędkości
        combined = torch.cat([cnn_features, speed_features], dim=1)
        logits = self.fc(combined)
        return logits