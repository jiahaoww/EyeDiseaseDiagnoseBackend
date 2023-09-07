import torch
from torchvision.models import swin_s, Swin_S_Weights
# https://pytorch.org/vision/master/models/swin_transformer.html
# Swin-T和Swin-S的复杂度分别与ResNet-50(DeiT-S)和ResNet-101相似
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class MyModel_swin_fundus(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a Swin_T backbone
        m2 = swin_s()
        m2.load_state_dict(torch.load('swin_s_weight.pth'))
        return_nodes2 = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images):
        x_features = self.body_2(images)
        output = self.fc(x_features['[batch, 768]'])
        return output