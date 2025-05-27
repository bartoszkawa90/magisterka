import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Residual block for stable deeper features ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(out + residual)

# --- Combined Actor + Critic base ---
class BaseNet(nn.Module):
    def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
        super().__init__()
        self.device = device

        # --- Image backbone (very similar to original) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            # two residual blocks at 64 channels
            ResidualBlock(64),
            ResidualBlock(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        # --- Velocity branch (unchanged size) ---
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU()
        )

        # these get overridden in Actor/Critic
        self.head = nn.Identity()

    def extract_features(self, x, speed):
        # image normalization
        x = x.to(self.device, dtype=torch.float32) / 255.0
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(x.size(0), -1)  # Flatten

        # what to do when speed considered
        if speed is None:
            speed = torch.zeros((x.size(0), 1), device=self.device)
        else:
            speed = speed.to(self.device, dtype=torch.float32)
            if speed.dim() == 1:
                speed = speed.unsqueeze(1)  # make sure to [B, 1]

        speed_features = self.speed_fc(speed)

        return torch.cat([cnn_features, speed_features], dim=1)

# --- Actor with custom head ---
class Actor(BaseNet):
    def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
        super().__init__(input_shape, actor_shape, device)
        self.left_head = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.LayerNorm(512), # may fasten learning process
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )
        self.forward_head = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.LayerNorm(512), # may fasten learning process
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )
        self.right_head = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.LayerNorm(512), # may fasten learning process
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )

        self.heads = nn.ModuleList([
            self.forward_head,
            self.right_head,
            self.left_head
        ])

    def forward(self, img, speed=None, manouver=None):
        feat = self.extract_features(img, speed)
        head = self.heads[int(torch.argmax(manouver))]
        return head(feat)

# --- Critic with custom head ---
class Critic(BaseNet):
    def __init__(self, input_shape, actor_shape, device=torch.device("cuda")):
        super().__init__(input_shape, actor_shape, device)
        self.left_head = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.LayerNorm(512), # may fasten learning process
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )
        self.forward_head = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.LayerNorm(512), # may fasten learning process
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )
        self.right_head = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 32, 512),
            nn.LayerNorm(512), # may fasten learning process
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, actor_shape)
        )

        self.heads = nn.ModuleList([
            self.forward_head,
            self.right_head,
            self.left_head
        ])

    def forward(self, img, speed=None, manouver=None):
        feat = self.extract_features(img, speed)
        head = self.heads[int(torch.argmax(manouver))]
        return head(feat)
