import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import os
from torch.utils import data
from DRModel.DRModel import MyModel_swin_fundus
from tqdm import tqdm
from collections import OrderedDict

transforms_ =transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

batch_size= 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(data.Dataset):
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
    
images = [[]]
dataset = Dataset(df_data = images, transform = transforms_)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
model = any

DR_NUM_CLASSES = 4

def init_model():
    global model
    model = MyModel_swin_fundus(DR_NUM_CLASSES)
    model = model.to(device)
    checkpoint = torch.load("swin_dr_grading.pth", map_location=torch.device("cuda:0"))
    
    prop_selected = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        prop_selected[name] = v
    model.load_state_dict(prop_selected)
    model.eval()

def updateDataLoader(imgPath_Array):
    global dataset
    global loader
    dataset.update(imgPath_Array)
    loader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

def getDRType(imgPath): 
    updateDataLoader([imgPath])
    for image in tqdm(loader):
        image = image.cuda(non_blocking=True)
        outputs = model(image)
        y_pred = []
        y_pred = y_pred + outputs.argmax(1).tolist()
    torch.cuda.empty_cache()
    return y_pred[0]

init_model()