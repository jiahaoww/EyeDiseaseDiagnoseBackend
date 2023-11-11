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
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


class jyk_dataset(data.Dataset):
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
        image = cv2.resize(image, (224, 224))

        if self.transform is not None:
            image = self.transform(image)
        return image


jyk_transforms_ = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])


transforms_ =transforms.Compose([
    transforms.Resize((384,)),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
])

batch_size= 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2

jyk_num_classes = 5

model = any

jyk_model = torchvision.models.resnet18(pretrained=False)

active_model = any

active_dataset = any

def update_active_model(type):
    global active_model
    global active_dataset
    if (type == 1):
        active_model = model
        active_dataset = test_data
    else:
        active_model = jyk_model
        active_dataset = jyk_data

def init_model():
    global model
    model = MyModel_single_fundus(num_classes)
    model = model.to(device)
    checkpoint = torch.load("swin_fundus_700.pth", map_location = torch.device("cuda:0"))
    
    prop_selected = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        prop_selected[name] = v
    model.load_state_dict(prop_selected)
    model.eval()


def init_jyk_model():
    global jyk_model
    num_features = jyk_model.fc.in_features
    jyk_model.fc = nn.Linear(num_features, jyk_num_classes)

    jyk_model = jyk_model.to(device)
    jyk_model.load_state_dict(torch.load("Net0.pth", map_location=torch.device("cuda:0")))
    jyk_model.eval()

# test_data = pd.read_csv('./test_Data.csv').values

images = [[]]
test_data = Mydataset(df_data = images, transform = transforms_)
jyk_data = jyk_dataset(df_data = images, transform = jyk_transforms_)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)


def updateDataLoader(imgPath_Array):
    global active_dataset
    global test_loader
    active_dataset.update(imgPath_Array)
    test_loader=DataLoader(active_dataset,batch_size=batch_size,shuffle=True)

init_model()
init_jyk_model()

def predict(imgPath): 
    updateDataLoader([imgPath])
    Y_Pred = []
    list = []
    for _, (batch_val) in enumerate(test_loader):
        pred_val = active_model(batch_val.to(device))
        pred_val = torch.softmax(pred_val, 1)
        list = pred_val.tolist()
        Y_Pred.append(torch.max(pred_val, 1)[1].cpu().numpy())
    torch.cuda.empty_cache()
    # Y_Pred = np.hstack(Y_Pred)
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

heatmap_model = torchvision.models.resnet18(pretrained=False)

def init_heatmap_model():
  global heatmap_model
  num_features = heatmap_model.fc.in_features
  heatmap_model.fc = nn.Linear(num_features, 5)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  heatmap_model = heatmap_model.to(device)
  heatmap_model.load_state_dict(torch.load("Net0.pth", map_location = device))
  heatmap_model.eval()

init_heatmap_model()

target_layers = [heatmap_model.layer4]

def generate_heatmap_image(image_path):
    rgb_img = cv2.imread(f'./image/{image_path}', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    cam_algorithm = methods[heatmap_method]
    with cam_algorithm(model = heatmap_model,
                    target_layers = target_layers,
                    use_cuda = True) as cam:

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
      cam.batch_size = 32
      grayscale_cam = cam(input_tensor=input_tensor,
                          targets=None,
                          aug_smooth=False,
                          eigen_smooth=False)

      # Here grayscale_cam has only one image in the batch
      grayscale_cam = grayscale_cam[0, :]

      cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
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
      name = os.path.splitext(image_path)[0]
      cv2.imwrite(f'{heatmap_path}/{name}.jpg', heatmap)
      torch.cuda.empty_cache()


      # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.

      # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(f'{heatmap_method}_cam.jpg', cam_image)