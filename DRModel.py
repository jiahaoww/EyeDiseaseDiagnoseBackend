import torch
from torchvision.models import swin_s, Swin_S_Weights
# https://pytorch.org/vision/master/models/swin_transformer.html
# Swin-T和Swin-S的复杂度分别与ResNet-50(DeiT-S)和ResNet-101相似
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


class MyModel_swin_OCT(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a Swin_T backbone
        m1 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        return_nodes1 = {
            'flatten': '[batch, 768]',
        }
        self.body_1 = create_feature_extractor(m1, return_nodes1)
        
        # train_nodes, eval_nodes = get_graph_node_names(swin_t())
        # print(train_nodes)
        
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images):
        x_features = self.body_1(images)
        # print(x_features1)
        # print(x_features1['[batch, 768]'].shape)
        # for k,v in x_features1.items():
        #     print(k)
        #     print(v.shape)
        # print(x_features1.shape)
        output = self.fc(x_features['[batch, 768]'])
        # x = self.body(x)
        # x = self.fpn(x)
        return output
    

class MyModel_swin_fundus(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # Get a Swin_T backbone
        
        m2 = swin_s()
        m2.load_state_dict(torch.load('tangwang_weight.pth'))
        return_nodes2 = {
            'flatten': '[batch, 768]',
        }
        self.body_2 = create_feature_extractor(m2, return_nodes2)
        
        # train_nodes, eval_nodes = get_graph_node_names(swin_t())
        # print(train_nodes)
        
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, images):
        x_features = self.body_2(images)
        # print(x_features1)
        # print(x_features1['[batch, 768]'].shape)
        # for k,v in x_features1.items():
        #     print(k)
        #     print(v.shape)
        # print(x_features1.shape)
        output = self.fc(x_features['[batch, 768]'])
        # x = self.body(x)
        # x = self.fpn(x)
        return output