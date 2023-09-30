#encoding=gbk
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import os
from torch.utils import data
from tqdm import tqdm
from collections import OrderedDict
from multiModel.multiModel import MyModel_swin_trans

transforms_ =transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

# OCT_FUNDUS
class OCT_FUNDUS_DATASET(data.Dataset):
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
        img_path1 = os.path.join(self.data_dir, img_name[0])
        img_path2 = os.path.join(self.data_dir, img_name[1])
        image1 = cv2.imread(img_path1)
        image1 = cv2.resize(image1, (224, 224))#256*256
        image2 = cv2.imread(img_path2)
        image2 = cv2.resize(image1, (224, 224))#256*256

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2
    
images = [[]]
dataset = OCT_FUNDUS_DATASET(df_data = images, transform = transforms_)
loader= DataLoader(dataset, batch_size = 1, shuffle=True)
model = any

NUM_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model():
    global model
    model = MyModel_swin_trans(NUM_CLASSES)
    model = model.to(device)
    dict = torch.load("./swin_trans_2000.pth", map_location = torch.device("cuda:0"))
    
    selected = OrderedDict()
    for k, v in dict.items():
        name = k[7:]  # remove `module.`
        selected[name] = v
    
    model.load_state_dict(selected, strict = True)
    model.eval()

def update_DataLoader(imgPath_Array):
    global dataset
    global loader
    dataset.update(imgPath_Array)
    loader = DataLoader(dataset, batch_size= 1, shuffle = True)

def get_OCT_FUNDUS_Type(imgPath): 
    update_DataLoader([imgPath])
    for image1, image2 in tqdm(loader):
        images1 = image1.cuda(non_blocking = True)
        images2 = image2.cuda(non_blocking = True)
        outputs = model(images1, images2)
        y_pred = []
        y_pred = y_pred + outputs.argmax(1).tolist()
    torch.cuda.empty_cache()
    return y_pred[0]


init_model()
