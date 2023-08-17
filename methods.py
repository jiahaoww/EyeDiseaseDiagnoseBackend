#encoding=gbk
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import os
from torch.utils import data
import torchvision
import numpy as np
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import datetime


def imageList():
    return os.listdir('./image')


class Mydataset(data.Dataset):
    def __init__(self,df_data, data_dir = '', transform = None):
        super().__init__()
        self.df = df_data
        self.data_dir = data_dir
        self.transform = transform

    def update(self, df_data):
        self.df = df_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idex):
        img_name = self.df[idex]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))#256*256

        if self.transform is not None:
            image = self.transform(image)
        return image



transforms_ =transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

batch_size= 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

model= torchvision.models.resnet18(pretrained=False)

def init_model():
    global model
    num_features=model.fc.in_features
    model.fc=nn.Linear(num_features,num_classes)

    model = model.to(device)
    model.load_state_dict(torch.load("Net0.pth",map_location=torch.device('cpu')))
    model.eval()

# test_data = pd.read_csv('./test_Data.csv').values

images = [
    ['../image/20210819170737613002.tif']
]
test_data= Mydataset(df_data = images,transform=transforms_)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True)


def updateDataLoader(imgPath_Array):
    global test_data
    global test_loader
    test_data.update(imgPath_Array)
    test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True)

init_model()

def predict(imgPath): 
    updateDataLoader([imgPath])
    Y_Pred = []
    list = []
    for _, (batch_val) in enumerate(test_loader):
        pred_val = model(batch_val.to(device))
        pred_val = torch.softmax(pred_val, 1)
        list = pred_val.tolist()
        Y_Pred.append(torch.max(pred_val, 1)[1].cpu().numpy())
    torch.cuda.empty_cache()

    # Y_Pred = np.hstack(Y_Pred)
    return list[0]