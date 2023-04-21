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

#创建文件索引和标签
# parts = ["train","val","test"]
# for part in parts:
#     dataset = pd.DataFrame()
#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/正常眼")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/正常眼/{train_file}",'label': 0}
#         dataset = dataset.append(new,ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/糖网")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/糖网/{train_file}",'label': 1}
#         dataset = dataset.append(new,ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/老黄")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/老黄/{train_file}", 'label': 2}
#         dataset = dataset.append(new, ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/静阻")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/静阻/{train_file}", 'label': 3}
#         dataset = dataset.append(new, ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/高度近视")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/高度近视/{train_file}", 'label': 4}
#         dataset = dataset.append(new, ignore_index=True)

#     trains = os.listdir(f"/home/Disk2/mixeddataset/{part}/青光眼")
#     for train_file in trains:
#         new = {'name': f"/home/Disk2/mixeddataset/{part}/青光眼/{train_file}", 'label': 5}
#         dataset = dataset.append(new, ignore_index=True)

#     dataset.to_csv(f"{part}_Data.csv",index=None)

# print("创建数据集成功")

#定义自己的数据集
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
        image = cv2.imread(img_path)#根据图片路径读取图片
        image = cv2.resize(image, (224, 224))#256*256

        if self.transform is not None:
            image = self.transform(image)
        return image



transforms_ =transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),#水平翻转
    # transforms.RandomRotation(10),#//随机旋转10度
    #transforms.Resize(224,224), #缩放图片，保持长宽比不变，最短边的长为224像素,
    # transforms.CenterCrop(224), #从中间切出 224*224的图片
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])  # 标准化至[-1,1]

# transforms_1 =transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.RandomHorizontalFlip(),#水平翻转
#     transforms.RandomRotation(10),#//随机旋转10度
#     #transforms.Resize(224,224), #缩放图片，保持长宽比不变，最短边的长为224像素,
#     # transforms.CenterCrop(224), #从中间切出 224*224的图片
#     transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
#     transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])  # 标准化至[-1,1]

# 进行数据增广
batch_size= 1

# train_data_dir = pd.read_csv('./train_Data.csv').values
# train_data=Mydataset(df_data = train_data_dir,transform=transforms_)#实例化训练数据
# train_data_aug=Mydataset(df_data = train_data_dir,transform=transforms_)#实例化训练数据

# train_data_aug=Mydataset(df_data=train_data,transform=transforms_1)
# print(train_data.type())
# train_data=np.vstack([train_data.df,train_data_aug.df])

# train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)#实例化训练集的DataLoader
# train_loader_aug=DataLoader(train_data_aug,batch_size=batch_size,shuffle=True)#实例化训练集的DataLoader


test_data = pd.read_csv('./test_Data.csv').values
test_data = [
    ['../image/20210819170737613002.tif']
]
test_data= Mydataset(df_data = test_data,transform=transforms_)#实例化训练数据
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True)#实例化训练集的DataLoader

# val_data = pd.read_csv('./val_Data.csv').values
# val_data=Mydataset(df_data = val_data,transform=transforms_)#实例化训练数据
# val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=True)#实例化训练集的DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate=0.0001#学习率
weight_decay=1e-6#权重衰减系数
episode_num =10#迭代次数
num_classes = 5


# model=torchvision.models.resnet18(pretrained=True)
# num_features=model.fc.in_features
# print(num_features)
# model.fc=nn.Linear(num_features,num_classes)
# model.eval()
# model=model.to(device)
# print(model)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)#adam优化器
# criterion = nn.CrossEntropyLoss()#交叉熵损失函数
# criterion.to(device)


# print("=====Training Phase=====")
# torch.cuda.empty_cache()
# for episode in range(episode_num):#迭代训练
#     print("===Episode {}/{}===".format(episode+1, episode_num))
#     model.zero_grad()
#     loss_list = list()
#     train_acc_list = []
#     for step, (batch_x, batch_y) in enumerate(train_loader):#在train_loader取出batch_size个数据
#         optimizer.zero_grad()
#         pred_y = model(batch_x.to(device))#前向传播得到预测值
#         # loss = criterion(predict, target)
#         loss = criterion(pred_y, batch_y.long().to(device))#计算损失值
#         loss.backward()#反向传播
#         optimizer.step()#更新模型参数
#         # total_loss += loss.item()
#         loss_list.append(loss.item()/batch_size)#记录历史损失
#         train_acc_list.append(accuracy_score(batch_y.cpu().numpy(), torch.max(pred_y, 1)[1].cpu().numpy()))
#         if((step+1)%10==0):#每5个step评估一次模型
#             model.eval()
#             val_acc = np.zeros(len(val_loader))
#             for val_step, (batch_val, target_val) in enumerate(val_loader):#从val_loader取出数据
#                 pred_val = model(batch_val.to(device))
#                 val_acc[val_step] = accuracy_score(target_val.cpu().numpy(), torch.max(pred_val, 1)[1].cpu().numpy())#计算准确率
#             print('Episode:', episode+1, '| Step:', step+1,"| Loss=", np.mean(np.array(loss_list)), "| Train Accuracy=", np.mean(np.array(train_acc_list)), "| Val Accuracy={:.3f}".format(np.mean(val_acc)))
#     for step, (batch_x, batch_y) in enumerate(train_loader_aug):  # 在train_loader取出batch_size个数据
#         optimizer.zero_grad()
#         pred_y = model(batch_x.to(device))  # 前向传播得到预测值
#         # loss = criterion(predict, target)
#         loss = criterion(pred_y, batch_y.long().to(device))  # 计算损失值
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新模型参数
#         # total_loss += loss.item()
#         loss_list.append(loss.item() / batch_size)  # 记录历史损失
#         train_acc_list.append(accuracy_score(batch_y.cpu().numpy(), torch.max(pred_y, 1)[1].cpu().numpy()))
#         if ((step + 1) % 10 == 0):  # 每5个step评估一次模型
#             model.eval()
#             val_acc = np.zeros(len(val_loader))
#             for val_step, (batch_val, target_val) in enumerate(val_loader):  # 从val_loader取出数据
#                 pred_val = model(batch_val.to(device))
#                 val_acc[val_step] = accuracy_score(target_val.cpu().numpy(),
#                                                    torch.max(pred_val, 1)[1].cpu().numpy())  # 计算准确率
#             print('Episode:', episode + 1, '| Step:', step + 1, "| Loss=", np.mean(np.array(loss_list)),
#                   "| Train Accuracy=", np.mean(np.array(train_acc_list)),
#                   "| Val Accuracy={:.3f}".format(np.mean(val_acc)))

# torch.save(model.state_dict(), 'Net0.pth')  # save net model and parameters保存模型

print("=====Testing Phase=====")
model=torchvision.models.resnet18(pretrained=False)
num_features=model.fc.in_features
model.fc=nn.Linear(num_features,num_classes)

model = model.to(device)
model.load_state_dict(torch.load("Net0.pth",map_location=torch.device('cpu')))#重新加载模型
model.eval()
val_acc = np.zeros(len(test_loader))
Y_Test = []
Y_Pred = []
for val_step, (batch_val) in enumerate(test_loader):#在测试集中取出数据进行预测
    pred_val = model(batch_val.to(device))#前向传播得到预测数据
    pred_val = torch.softmax(pred_val, 1)
    print(pred_val)
    # Y_Test.append(target_val.cpu().numpy())#记录所有的实际值
    Y_Pred.append(torch.max(pred_val, 1)[1].cpu().numpy())#记录所有的预测值

# Y_Test = np.hstack(Y_Test)
Y_Pred = np.hstack(Y_Pred)
print(Y_Pred)



# # cm = confusion_matrix(Y_Test, Y_Pred)#输出分类混淆矩阵
# print(classification_report(Y_Test, Y_Pred))#输出分类评估报告
# # print(cm)

# classes=['正常眼','糖网','老黄','静阻','高度近视']
# cm = confusion_matrix(Y_Test, Y_Pred, labels=range(len(classes)))
# acc_avg, acc, f1_macro, f1, sensitivity, specificity, precision = evaluate_metrics(cm,classes)
# #print('Average Accuracy -> {:>6.4f}, Macro F1 -> {:>6.4f} and Cohen\'s Kappa -> {:>6.4f} on test set'.format(acc_avg, f1_macro, ck_score))

# for index_ in range(len(classes)):
#     print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, F1 : {:1.4f} Accuracy: {:1.4f}".format(
#             classes[index_], sensitivity[index_], specificity[index_], precision[index_], f1[index_], acc[index_]))

