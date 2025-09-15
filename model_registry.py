import torch
import torch.nn as nn

class ShallowEncoder(nn.Module):
    def __init__(self, input_size=256):
        super().__init__()
        hidden_size = input_size * 2
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

class ShallowDecoder(nn.Module):
    def __init__(self, output_size=256):
        super().__init__()
        hidden_size = output_size * 2
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# --- Deep Model (5 layers) ---

class DeepEncoder(nn.Module):
    def __init__(self, input_size=256):
        super().__init__()
        hidden_size = input_size * 2
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
    def __init__(self, output_size=256):
        super().__init__()
        hidden_size = output_size * 2
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

# --- Model Registry ---

MODEL_REGISTRY = {
    "shallow": {
        "input_dim": 256,
        "encoder": lambda: ShallowEncoder(256),
        "decoder": lambda: ShallowDecoder(256),
        "savedir": "shallow",
        "process":"QCD"
    },
    "deep": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "deep",
        "process":"QCD"
    },
    "deep_ttbar": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "deep_ttbar",
        "process":"TTto4Q"
    },
}
