import torch
from torchvision.models import swin_b
from torchvision.models.feature_extraction import create_feature_extractor

class MyModel_single_fundus(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        m1 = swin_b()
        m1.load_state_dict(torch.load('swin_b_weight.pth'))
        return_nodes2 = {
            'flatten': '[batch, 1024]',
        }
        self.body_2 = create_feature_extractor(m1, return_nodes2)
        self.fc = torch.nn.Linear(1024, num_classes)

    def forward(self, images):
        x_features = self.body_2(images)
        output = self.fc(x_features['[batch, 1024]'])
        return output