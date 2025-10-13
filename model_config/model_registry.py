import torch
import torch.nn as nn

class ShallowEncoder(nn.Module):
    def __init__(self, input_size=256,width_multiplier=2):
        super().__init__()
        hidden_size = input_size * width_multiplier
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

class ShallowDecoder(nn.Module):
    def __init__(self, output_size=256,width_multiplier=2):
        super().__init__()
        hidden_size = output_size * width_multiplier
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# --- Deep Model (5 layers) ---

class DeepEncoder(nn.Module):
    def __init__(self, input_size=256,width_multiplier=2):
        super().__init__()
        hidden_size = input_size * width_multiplier
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
    def __init__(self, output_size=256,width_multiplier=2):
        super().__init__()
        hidden_size = output_size * width_multiplier
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
        "savedir": "models/shallow",
        "process":"QCD"
    },
    "deep": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "models/deep_new_training",
        "process":"QCD"
    },
    "feat16_encoder64_deep_qcd": {
        "input_dim": 16,
        "encoder": lambda: DeepEncoder(16,4),
        "decoder": lambda: DeepDecoder(16,4),
        "savedir": "models/feat16_encoder64_deep_qcd",
        "process":"QCD"
    },
    "feat16_encoder64_shallow_qcd": {
        "input_dim": 16,
        "encoder": lambda: ShallowEncoder(16,4),
        "decoder": lambda: ShallowDecoder(16,4),
        "savedir": "models/feat16_encoder64_shallow_qcd",
        "process":"QCD"
    },
    "feat32_encoder128_deep_qcd": {
        "input_dim": 32,
        "encoder": lambda: DeepEncoder(32,4),
        "decoder": lambda: DeepDecoder(32,4),
        "savedir": "models/feat32_encoder128_deep_qcd",
        "process":"QCD"
    },
    "deep_ttbar": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "models/deep_ttbar",
        "process":"TTto4Q"
    },
    "feat2_encoder32_shallow_ttbar": {
        "input_dim": 2,
        "encoder": lambda: ShallowEncoder(2,16),
        "decoder": lambda: ShallowDecoder(2,16),
        "savedir": "models/feat2_encoder32_shallow_ttbar",
        "process":"TTto4Q"
    },  
    "feat16_encoder128_shallow_ttbar": {
        "input_dim": 16,
        "encoder": lambda: ShallowEncoder(16,8),
        "decoder": lambda: ShallowDecoder(16,8),
        "savedir": "models/feat16_encoder128_shallow_ttbar",
        "process":"TTto4Q"
    },
    "feat64_encoder256_shallow_ttbar": {
        "input_dim": 64,
        "encoder": lambda: ShallowEncoder(64,4),
        "decoder": lambda: ShallowDecoder(64,4),
        "savedir": "models/feat64_encoder256_shallow_ttbar",
        "process":"TTto4Q"
    },  
    "feat128_encoder512_shallow_ttbar": {
        "input_dim": 128,
        "encoder": lambda: ShallowEncoder(128,4),
        "decoder": lambda: ShallowDecoder(128,4),
        "savedir": "models/feat128_encoder512_shallow_ttbar",
        "process":"TTto4Q"
    },  
    "feat128_encoder1024_shallow_ttbar": {
        "input_dim": 128,
        "encoder": lambda: ShallowEncoder(128,8),
        "decoder": lambda: ShallowDecoder(128,8),
        "savedir": "models/feat128_encoder1024_shallow_ttbar",
        "process":"TTto4Q"
    },    
    "feat4_encoder32_deep_qcd": {
        "input_dim": 4,
        "encoder": lambda: DeepEncoder(4,8),
        "decoder": lambda: DeepDecoder(4,8),
        "savedir": "models/feat4_encoder32_deep_qcd",
        "process":"QCD"
    },    
    "feat4_encoder32_deep_bqq": {
        "input_dim": 4,
        "encoder": lambda: DeepEncoder(4,8),
        "decoder": lambda: DeepDecoder(4,8),
        "savedir": "models/feat4_encoder32_deep_bqq",
        "process":"Top_bqq"
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
    "deep_qcd_tt": {
        "input_dim": 256,
        "encoder": lambda: DeepEncoder(256),
        "decoder": lambda: DeepDecoder(256),
        "savedir": "models/deep_qcd_tt",
        "process":"QCD"
    },
}
