import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, board_size=3, input_planes=2, filters=256, blocks=10, action_size=9):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.conv = nn.Conv2d(input_planes, filters, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.res_blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(blocks)])
        # policy head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)
        # value head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, filters)
        self.value_fc2 = nn.Linear(filters, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.res_blocks(out)
        # policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        log_p = F.log_softmax(p, dim=1)
        # value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return log_p, v.squeeze(-1)