import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F

class EDLModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EDLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # ✅ Correct
        self.fc2 = nn.Linear(1024, 512)  # ✅ Increased size from 256 to 512
        self.fc3 = nn.Linear(512, 256)  # ✅ Now correctly takes 512 inputs
        self.fc4 = nn.Linear(256, num_classes)  # ✅ Outputs final class predictions
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x, activation='softmax'):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if activation == 'softmax':
            x = F.softmax(x, dim=1)
        elif activation == 'softplus':
            x = F.softplus(x)
        return x
