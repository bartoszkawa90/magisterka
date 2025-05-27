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
    def __init__(self, input_shape, n_actions, device=torch.device("cuda")):
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

        # --- Learnable embedding for 3-way maneuver ---
        self.maneuver_emb = nn.Embedding(3, 32)

        # --- Combined feature size: 256*4*4 + 32 + 32 = 4352 ---
        total_feat = 256 * 4 * 4 + 32 + 32

        # these get overridden in Actor/Critic
        self.head = nn.Identity()

    def extract_features(self, img, speed, maneuver):
        # image → features
        x = img.to(self.device, torch.float32) / 255.0
        x = self.cnn(x).view(x.size(0), -1)

        # speed → features
        if speed is None:
            speed = torch.zeros(x.size(0), 1, device=self.device)
        speed = speed.to(self.device, torch.float32).unsqueeze(-1)
        sp = self.speed_fc(speed)

        # maneuver → features
        idx = maneuver.to(self.device) if maneuver is not None else torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        mg = self.maneuver_emb(idx)

        return torch.cat([x, sp, mg], dim=1)

# --- Actor with simple MLP head ---
class CombinedActor(BaseNet):
    def __init__(self, input_shape, n_actions, device=torch.device("cuda")):
        super().__init__(input_shape, n_actions, device)
        total_feat = 256 * 4 * 4 + 32 + 32
        self.head = nn.Sequential(
            nn.Linear(total_feat, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_actions)
        )

    def forward(self, img, speed=None, maneuver=None):
        feat = self.extract_features(img, speed, maneuver)
        return self.head(feat)

# --- Dueling Critic: separates state-value and action advantages ---
class CombinedCritic(BaseNet):
    def __init__(self, input_shape, n_actions, device=torch.device("cuda")):
        super().__init__(input_shape, n_actions, device)
        total_feat = 256 * 4 * 4 + 32 + 32

        self.value_head = nn.Sequential(
            nn.Linear(total_feat, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(total_feat, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_actions)
        )

    def forward(self, img, speed=None, maneuver=None):
        feat = self.extract_features(img, speed, maneuver)
        V = self.value_head(feat)                   # [B, 1]
        A = self.adv_head(feat)                     # [B, n_actions]
        A_mean = A.mean(dim=1, keepdim=True)        # [B, 1]
        return V + (A - A_mean)                     # dueling aggregation
