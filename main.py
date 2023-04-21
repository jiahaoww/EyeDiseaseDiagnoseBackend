#encoding=gbk
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import pandas as pd
import cv2
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score,classification_report
import numpy as np
import PIL.Image as Image
from torchvision import transforms as tfs
import torchvision
import torch.nn as nn
from evaluate_metrics import evaluate_metrics
import numpy as np


def dropout(x,drop_prob):
    x=x.float()
    assert 0<=drop_prob<=1
    keep_prob=1-drop_prob
    if keep_prob==0:
        return torch.zeros_like(x)
    mask=(torch.randn(x.shape)<keep_prob).float()
    return mask*x/keep_prob

#�����ļ������ͱ�ǩ
# parts = ["train","val","test"]
# for part in parts:
#     dataset = pd.DataFrame()
#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/������")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/������/{train_file}",'label': 0}
#         dataset = dataset.append(new,ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/����")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/����/{train_file}",'label': 1}
#         dataset = dataset.append(new,ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/�ϻ�")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/�ϻ�/{train_file}", 'label': 2}
#         dataset = dataset.append(new, ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/����")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/����/{train_file}", 'label': 3}
#         dataset = dataset.append(new, ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/�߶Ƚ���")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/�߶Ƚ���/{train_file}", 'label': 4}
#         dataset = dataset.append(new, ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/�����")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/�����/{train_file}", 'label': 5}
#         dataset = dataset.append(new, ignore_index=True)

#     dataset.to_csv(f"{part}_Data.csv",index=None)

# print("�������ݼ��ɹ�")

#�����Լ������ݼ�
class Mydataset(data.Dataset):
    def __init__(self,df_data, data_dir = '', transform = None):
        super().__init__()
        self.df = df_data
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idex):
        img_name,  = self.df[idex]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)#����ͼƬ·����ȡͼƬ
        image = cv2.resize(image, (224, 224))#256*256

        if self.transform is not None:
            image = self.transform(image)
        return image



transforms_ =transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),#ˮƽ��ת
    # transforms.RandomRotation(10),#//�����ת10��
    #transforms.Resize(224,224), #����ͼƬ�����ֳ���Ȳ��䣬��̱ߵĳ�Ϊ224����,
    # transforms.CenterCrop(224), #���м��г� 224*224��ͼƬ
    transforms.ToTensor(),  # ��ͼƬת��ΪTensor,��һ����[0,1]
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])  # ��׼����[-1,1]

# transforms_1 =transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.RandomHorizontalFlip(),#ˮƽ��ת
#     transforms.RandomRotation(10),#//�����ת10��
#     #transforms.Resize(224,224), #����ͼƬ�����ֳ���Ȳ��䣬��̱ߵĳ�Ϊ224����,
#     # transforms.CenterCrop(224), #���м��г� 224*224��ͼƬ
#     transforms.ToTensor(),  # ��ͼƬת��ΪTensor,��һ����[0,1]
#     transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])  # ��׼����[-1,1]

# ������������
batch_size= 1

# train_data_dir = pd.read_csv('./train_Data.csv').values
# train_data=Mydataset(df_data = train_data_dir,transform=transforms_)#ʵ����ѵ������
# train_data_aug=Mydataset(df_data = train_data_dir,transform=transforms_)#ʵ����ѵ������

# train_data_aug=Mydataset(df_data=train_data,transform=transforms_1)
# print(train_data.type())
# train_data=np.vstack([train_data.df,train_data_aug.df])

# train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)#ʵ����ѵ������DataLoader
# train_loader_aug=DataLoader(train_data_aug,batch_size=batch_size,shuffle=True)#ʵ����ѵ������DataLoader


test_data = pd.read_csv('./test_Data.csv').values
test_data = [
    ['../image/20210819170737613002.tif']
]
test_data= Mydataset(df_data = test_data,transform=transforms_)#ʵ����ѵ������
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True)#ʵ����ѵ������DataLoader

# val_data = pd.read_csv('./val_Data.csv').values
# val_data=Mydataset(df_data = val_data,transform=transforms_)#ʵ����ѵ������
# val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=True)#ʵ����ѵ������DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate=0.0001#ѧϰ��
weight_decay=1e-6#Ȩ��˥��ϵ��
episode_num =10#��������
num_classes = 5


# model=torchvision.models.resnet18(pretrained=True)
# num_features=model.fc.in_features
# print(num_features)
# model.fc=nn.Linear(num_features,num_classes)
# model.eval()
# model=model.to(device)
# print(model)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)#adam�Ż���
# criterion = nn.CrossEntropyLoss()#��������ʧ����
# criterion.to(device)


# print("=====Training Phase=====")
# torch.cuda.empty_cache()
# for episode in range(episode_num):#����ѵ��
#     print("===Episode {}/{}===".format(episode+1, episode_num))
#     model.zero_grad()
#     loss_list = list()
#     train_acc_list = []
#     for step, (batch_x, batch_y) in enumerate(train_loader):#��train_loaderȡ��batch_size������
#         optimizer.zero_grad()
#         pred_y = model(batch_x.to(device))#ǰ�򴫲��õ�Ԥ��ֵ
#         # loss = criterion(predict, target)
#         loss = criterion(pred_y, batch_y.long().to(device))#������ʧֵ
#         loss.backward()#���򴫲�
#         optimizer.step()#����ģ�Ͳ���
#         # total_loss += loss.item()
#         loss_list.append(loss.item()/batch_size)#��¼��ʷ��ʧ
#         train_acc_list.append(accuracy_score(batch_y.cpu().numpy(), torch.max(pred_y, 1)[1].cpu().numpy()))
#         if((step+1)%10==0):#ÿ5��step����һ��ģ��
#             model.eval()
#             val_acc = np.zeros(len(val_loader))
#             for val_step, (batch_val, target_val) in enumerate(val_loader):#��val_loaderȡ������
#                 pred_val = model(batch_val.to(device))
#                 val_acc[val_step] = accuracy_score(target_val.cpu().numpy(), torch.max(pred_val, 1)[1].cpu().numpy())#����׼ȷ��
#             print('Episode:', episode+1, '| Step:', step+1,"| Loss=", np.mean(np.array(loss_list)), "| Train Accuracy=", np.mean(np.array(train_acc_list)), "| Val Accuracy={:.3f}".format(np.mean(val_acc)))
#     for step, (batch_x, batch_y) in enumerate(train_loader_aug):  # ��train_loaderȡ��batch_size������
#         optimizer.zero_grad()
#         pred_y = model(batch_x.to(device))  # ǰ�򴫲��õ�Ԥ��ֵ
#         # loss = criterion(predict, target)
#         loss = criterion(pred_y, batch_y.long().to(device))  # ������ʧֵ
#         loss.backward()  # ���򴫲�
#         optimizer.step()  # ����ģ�Ͳ���
#         # total_loss += loss.item()
#         loss_list.append(loss.item() / batch_size)  # ��¼��ʷ��ʧ
#         train_acc_list.append(accuracy_score(batch_y.cpu().numpy(), torch.max(pred_y, 1)[1].cpu().numpy()))
#         if ((step + 1) % 10 == 0):  # ÿ5��step����һ��ģ��
#             model.eval()
#             val_acc = np.zeros(len(val_loader))
#             for val_step, (batch_val, target_val) in enumerate(val_loader):  # ��val_loaderȡ������
#                 pred_val = model(batch_val.to(device))
#                 val_acc[val_step] = accuracy_score(target_val.cpu().numpy(),
#                                                    torch.max(pred_val, 1)[1].cpu().numpy())  # ����׼ȷ��
#             print('Episode:', episode + 1, '| Step:', step + 1, "| Loss=", np.mean(np.array(loss_list)),
#                   "| Train Accuracy=", np.mean(np.array(train_acc_list)),
#                   "| Val Accuracy={:.3f}".format(np.mean(val_acc)))

# torch.save(model.state_dict(), 'Net0.pth')  # save net model and parameters����ģ��

print("=====Testing Phase=====")
model=torchvision.models.resnet18(pretrained=False)
num_features=model.fc.in_features
model.fc=nn.Linear(num_features,num_classes)

model = model.to(device)
model.load_state_dict(torch.load("Net0.pth",map_location=torch.device('cpu')))#���¼���ģ��
model.eval()
val_acc = np.zeros(len(test_loader))
Y_Test = []
Y_Pred = []
for val_step, (batch_val) in enumerate(test_loader):#�ڲ��Լ���ȡ�����ݽ���Ԥ��
    pred_val = model(batch_val.to(device))#ǰ�򴫲��õ�Ԥ������
    pred_val = torch.softmax(pred_val, 1)
    print(pred_val)
    # Y_Test.append(target_val.cpu().numpy())#��¼���е�ʵ��ֵ
    Y_Pred.append(torch.max(pred_val, 1)[1].cpu().numpy())#��¼���е�Ԥ��ֵ

# Y_Test = np.hstack(Y_Test)
Y_Pred = np.hstack(Y_Pred)
print(Y_Pred)



# # cm = confusion_matrix(Y_Test, Y_Pred)#��������������
# print(classification_report(Y_Test, Y_Pred))#���������������
# # print(cm)

# classes=['������','����','�ϻ�','����','�߶Ƚ���']
# cm = confusion_matrix(Y_Test, Y_Pred, labels=range(len(classes)))
# acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm,classes)
# #print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))

# for index_ in range(len(classes)):
#     print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
#             classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))

