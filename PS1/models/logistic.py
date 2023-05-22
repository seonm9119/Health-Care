import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, dim_in, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs