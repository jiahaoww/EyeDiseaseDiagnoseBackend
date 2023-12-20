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

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from SingleModel.SingleModel import MyModel_single_fundus

from collections import OrderedDict

from PIL import Image

import torch
import models_vit
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import time


class fundusModel:
    def __init__(self, dataset, dataloader, net):
        self.dataset = dataset
        self.dataloader = dataloader
        self.net = net


class fundusDataset(data.Dataset):
    def __init__(self, df_data, data_dir = '', transform = None):
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
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    

class SatDataset(data.Dataset):
    def __init__(self, df_data, data_dir = '', transform = None):
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
        image = cv2.resize(image, (224, 224))
        if self.transform is not None:
            image = self.transform(image)
        return image  
    

def init_mae_model(num_classes, model_path):
    # df_data
    images = [[]]

    # transform
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    size = 256

    transform = transforms.Compose([
        transforms.Resize(size, interpolation = transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # dataset
    dataset = fundusDataset(df_data = images, transform = transform)

    # dataloader
    global batch_size
    dataloader = DataLoader(dataset, batch_size, shuffle = True)

    # net

    model = models_vit.__dict__['vit_large_patch16'](
        num_classes = num_classes,
        drop_path_rate = 0.2,
        global_pool = True,
    )
    # load RETFound weights
    checkpoint = torch.load(model_path, map_location = torch.device(location))
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
       if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
           print(f"Removing key {k} from pretrained checkpoint")
           del checkpoint_model[k]
    #interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=True)
    
    model = model.to(device)
    model.eval()

    return fundusModel(dataset, dataloader, model)


def init_swin_model(num_classes):
    # df_data
    images = [[]]

    # transform
    transform = transforms.Compose([
        transforms.Resize((384,)),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
    ])

    # dataset
    dataset = fundusDataset(df_data = images, transform = transform)

    # dataloader
    global batch_size
    dataloader = DataLoader(dataset, batch_size, shuffle = True)

    # net
    model = MyModel_single_fundus(num_classes)
    model = model.to(device)
    checkpoint = torch.load("swin_nolaohuang_1000.pth", map_location = torch.device("cuda:0"))
    
    prop_selected = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        prop_selected[name] = v
    model.load_state_dict(prop_selected)
    model.eval()
    return fundusModel(dataset, dataloader, model)

def init_sat_model():
    # df_data
    images = [[]]

    # transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    # dataset
    dataset = SatDataset(df_data = images, transform = transform)

    # dataloader
    global batch_size
    dataloader = DataLoader(dataset, batch_size, shuffle = True)

    # net
    num_classes = 5
    model = torchvision.models.resnet18(pretrained = False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    
    model.load_state_dict(torch.load("Net0.pth", map_location = torch.device(location)))

    model.eval()

    return fundusModel(dataset, dataloader, model)


batch_size = 1

useGPU = True

#useGPU = False

device = torch.device("cuda" if useGPU == True else "cpu")

location = "cuda:0" if useGPU == True else "cpu"

models = [init_mae_model(2, './checkpoint-best-normal.pth'), init_mae_model(2, './checkpoint-best-8468.pth'), init_mae_model(6, './checkpoint-best-6.pth')]
#models = [init_swin_model(5), init_mae_model(6, './checkpoint-best-6.pth'), init_sat_model(), init_mae_model(2, './checkpoint-best-8468.pth')]

active_model = models[0]

def update_active_model(index):
    global active_model
    active_model = models[index]


model_exec_count = 0

def predict(imgPath):
    start = time.time()
    global model_exec_count 
    model_exec_count += 1
    active_model.dataset.update([imgPath])
    list = []
    for _, (batch_val) in enumerate(active_model.dataloader):
        pred_val = active_model.net(batch_val.to(device))
        pred_val = torch.softmax(pred_val, 1)
        list = pred_val.tolist()
    
    torch.cuda.empty_cache()
    end = time.time()
    print(end - start)
    return list[0]

#heatmap part

heatmap_path = './heatmap_image/'

heatmap_method = 'gradcam'

methods = \
    {"gradcam": GradCAM,
      "hirescam": HiResCAM,
      "scorecam": ScoreCAM,
      "gradcam++": GradCAMPlusPlus,
      "ablationcam": AblationCAM,
      "xgradcam": XGradCAM,
      "eigencam": EigenCAM,
      "eigengradcam": EigenGradCAM,
      "layercam": LayerCAM,
      "fullgrad": FullGrad,
      "gradcamelementwise": GradCAMElementWise}

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_heatmap_image(image_path):

    rgb_img = cv2.imread(f'./image/{image_path}', 1)[:, :, ::-1]
    width = rgb_img.shape[0]
    height = rgb_img.shape[1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img,
    #                               mean=[0.485, 0.456, 0.406],
    #                               std=[0.229, 0.224, 0.225])
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
    cam_algorithm = methods[heatmap_method]

    target_layers = [active_model.net.blocks[-1].norm1]

    with cam_algorithm(model = active_model.net,
                    target_layers = target_layers,
                    use_cuda = True,
                    reshape_transform = reshape_transform) as cam:

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
      cam.batch_size = 32
      grayscale_cam = cam(input_tensor=input_tensor,
                          targets=None,
                          aug_smooth=False,
                          eigen_smooth=False)

      # Here grayscale_cam has only one image in the batch
      grayscale_cam = grayscale_cam[0, :]

      cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
      cam_image = cv2.resize(cam_image, (height, width))

      heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
      heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


      # for i in range(grayscale_cam.shape[0]):
      #       for j in range(grayscale_cam.shape[1]):
      #           if grayscale_cam[i, j] < 0.5:
      #               heatmap[i, j] = [0, 0, 0]

      # for i in range(grayscale_cam.shape[0]):
      #       for j in range(grayscale_cam.shape[1]):
      #           if grayscale_cam[i, j] < 0.5:
      #               heatmap[i, j] = [0, 0, 0]

      #heatmap = [0, 0, 0] if grayscale_cam < 0.5 else heatmap
      for i in range(3):
          heatmap[:,:,i] = np.where(grayscale_cam < 0.4, 0, heatmap[:,:,i])
      heatmap = np.float32(heatmap)
      heatmap = cv2.resize(heatmap, (height, width))
      name = os.path.splitext(image_path)[0]
      cv2.imwrite(f'{heatmap_path}/{name}.jpg', cam_image)
      torch.cuda.empty_cache()


      # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.

      # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(f'{heatmap_method}_cam.jpg', cam_image)