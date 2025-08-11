import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .config import Config


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.board_size = config.board_size
        self.action_size = config.action_size
        self.use_checkpointing = config.device != 'cpu' and config.blocks > 6
        
        # Backbone
        self.conv = nn.Conv2d(config.input_planes, config.filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(config.filters)
        self.res_blocks = nn.Sequential(*[ResidualBlock(config.filters) for _ in range(config.blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(config.filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * config.board_size * config.board_size, config.action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(config.filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_size * config.board_size, config.filters)
        self.value_fc2 = nn.Linear(config.filters, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        out = F.relu(self.bn(self.conv(x)))
        
        # Use gradient checkpointing for memory efficiency on GPU
        if self.use_checkpointing and self.training:
            out = checkpoint(self.res_blocks, out)
        else:
            out = self.res_blocks(out)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        log_p = F.log_softmax(p, dim=1)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return log_p, v.squeeze(-1)