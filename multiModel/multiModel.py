import torch
from torchvision.models import swin_t
from torchvision.models.feature_extraction import create_feature_extractor
from multiModel.cross_model_attention import Transformer

class MyModel_swin_trans(torch.nn.Module):
    def __init__(self,  num_classes):
        super().__init__()
        m1 = swin_t()
        m1.load_state_dict(torch.load('swin_t_weight.pth'))
        return_nodes1 = {
            'flatten': '[batch, 768]',
        }
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        m2 = swin_t()
        m2.load_state_dict(torch.load('swin_t_weight.pth'))
        return_nodes2 = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        self.Transformer_1 = Transformer(768)
        self.Transformer_2 = Transformer(768)
        self.fc_3072 = torch.nn.Linear(768 * 4, num_classes)

    def forward(self, images1, images2):
        x_features1 = self.body_1(images1)
        x_features2 = self.body_2(images2)
        x12 = self.Transformer_1(x_features1['[batch, 768]'],x_features2['[batch, 768]'])
        x21 = self.Transformer_2(x_features2['[batch, 768]'],x_features1['[batch, 768]'])
        x_feature = torch.cat((x12,x21,x_features1['[batch, 768]'],x_features2['[batch, 768]']),axis=1)
        output = self.fc_3072(x_feature)
        return output