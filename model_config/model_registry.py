import torch
import torch.nn as nn

class LatentNoiseWrapper(nn.Module):
    def __init__(self, encoder, noise_std=0.1):
        super().__init__()
        self.encoder = encoder
        self.noise_std = noise_std

    def forward(self, x):
        z = self.encoder(x)
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
        return z

class ShallowEncoder(nn.Module):
    def __init__(self, input_size=256,hidden_size=256):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

class ShallowDecoder(nn.Module):
    def __init__(self, output_size=256,hidden_size=256):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# --- Deep Model (5 layers) ---

class DeepEncoder(nn.Module):
    def __init__(self, input_size=256,hidden_size=256):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        return x

class DeepDecoder(nn.Module):
    def __init__(self, output_size=256,hidden_size=256):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        return self.layer5(x)


class BottleneckEncoder(nn.Module):
    def __init__(self, input_size=16, latent_size=4, hidden_size=16):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class BottleneckDecoder(nn.Module):
    def __init__(self, output_size=16, latent_size=4, hidden_size=16):
        super().__init__()
        self.layer1 = nn.Linear(latent_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class DeepEncoderBounded(nn.Module):
    """
    Deep encoder with bounded latent output using tanh.
    Prevents runaway latent magnitudes that destabilize reconstruction energy.
    """
    def __init__(self, input_size=256, hidden_size=256, latent_scale=1.0):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)

        self.latent_scale = latent_scale

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))

        x = torch.tanh(x * self.latent_scale)
        return x


# --- Model Registry ---

MODEL_REGISTRY = {
    "feat16_encoder64_deep_qcd": {
        "input_dim": 16,
        "encoder": lambda: DeepEncoder(16,64),
        "decoder": lambda: DeepDecoder(16,64),
        "savedir": "models/feat16_encoder64_deep_qcd",
        "process":"QCD"
    },
    "feat16_encoder12_bottleneck_qcd": {
        "input_dim": 16,
        "encoder": lambda: BottleneckEncoder(input_size=16, latent_size=12, hidden_size=32),
        "decoder": lambda: BottleneckDecoder(output_size=16, latent_size=12, hidden_size=32),
        "savedir": "models/feat16_encoder12_bottleneck_qcd",
        "process": "QCD"
    },
    "deep_ttbar": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "models/deep_ttbar",
        "process":"TTto4Q"
    },    
    "feat4_encoder32_deep_qcd": {
        "input_dim": 4,
        "encoder": lambda: DeepEncoder(input_size=4,hidden_size=32),
        "decoder": lambda: DeepDecoder(output_size=4,hidden_size=32),
        "savedir": "models/feat4_encoder32_deep_qcd",
        "process":"QCD"
    },
    "feat4_encoder4_deep_qcd": {
        "input_dim": 4,
        "encoder": lambda: DeepEncoder(input_size=4,hidden_size=4),
        "decoder": lambda: DeepDecoder(output_size=4,hidden_size=4),
        "savedir": "models/feat4_encoder4_deep_qcd",
        "process":"QCD"
    },
    "deep_qcd": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "models/deep_qcd",
        "process":"QCD"
    },
    "shallow_qcd": {
        "input_dim": 256,
        "encoder": lambda: ShallowEncoder(256),
        "decoder": lambda: ShallowDecoder(256),
        "savedir": "models/shallow_qcd",
        "process":"QCD"
    },
    "shallow2_encoder32_qcd": {
        "input_dim": 2,
        "encoder": lambda: ShallowEncoder(input_size=2,hidden_size=32),
        "decoder": lambda: ShallowDecoder(output_size=2,hidden_size=32),
        "savedir": "models/shallow2_encoder32_qcd",
        "process":"QCD"
    }
}
