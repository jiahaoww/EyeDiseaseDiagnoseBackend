import methods
import os
import numpy as np
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix
import time
import GPUtil
import matplotlib.pyplot as plt
import threading

def get_max_index(a):
   max_p = max(a)
   return a.index(max_p)

disease_abbr_6 = ['正常', '糖网', '老黄', '青光眼', '静阻', '高度近视']

correct = 0
incorrect = 0

y_label = []
y_pred = []

sensitivity = []
specificity = []
precision = []
accuracy = []
F1_score = []
kappa = []

# 整体6分类
def make_predict_1(path):
   methods.update_active_model(2)
   list = methods.predict(path)
   idx = get_max_index(list)
   return idx


# 先按正常、有病2分类，有病的再走后续
def make_predict_2(path):
   methods.update_active_model(0)
   list = methods.predict(path)
   idx = get_max_index(list)
   if (idx == 0):
      # 正常
      return idx
   else:
      methods.update_active_model(2)
      list = methods.predict(path)
      idx = get_max_index(list)
      return idx


# 先按正常、有病2分类，有病的再看是不是糖网
def make_predict_3(path):
   methods.update_active_model(0)
   list = methods.predict(path)
   idx = get_max_index(list)
   if (idx == 0):
      # 正常
      return idx
   else:
      methods.update_active_model(1)
      list = methods.predict(path)
      idx = get_max_index(list)
      if (idx == 0):
         return 1
      else:
         methods.update_active_model(2)
         list = methods.predict(path)
         idx = get_max_index(list)
         return idx


def calculate_one_class(folder, type):
   global correct
   global incorrect
   file_list = []
   for root, _, files in os.walk(folder):
      for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
   
   total = len(file_list)
   count = 0

   for file in file_list:
      res = make_predict_3(file)
      y_label.append(type)
      y_pred.append(res)
      if (res == type):
         count += 1
         correct += 1
      else:
         incorrect += 1


def calculate():
   home_dir = 'D:/fundus_test/'
   backup = ['zhengchang', 'tangwang', 'lh_s', 'qingguangyan', 'jingzu', 'jinshi']
   class_dir_list = backup
   num_class = 6
   id = 0
   start_time = time.time()
   for class_dir in class_dir_list:
      dir_path = home_dir + class_dir
      calculate_one_class(dir_path, id)
      id += 1
   end_time = time.time()
   print("elapsed_time: ", end_time - start_time)
   confusion_matrix = multilabel_confusion_matrix(np.array(y_label), np.array(y_pred), labels = [i for i in range(num_class)])
   for i in range(1, confusion_matrix.shape[0]):
      mat = confusion_matrix[i]
      sen = 1.0 * mat[1, 1] / (mat[1, 1] + mat[1, 0])
      spe = 1.0 * mat[0, 0] / (mat[0, 0] + mat[0, 1])
      pre = 1.0 * mat[1, 1] / (mat[1, 1] + mat[0, 1])
      acc = 1.0 * (mat[0, 0] + mat[1, 1]) / np.sum(mat)
      f1 = 2.0 * sen * pre / (sen + pre)
      pe = 1.0 * ((mat[1, 1] + mat[1, 0]) * (mat[1, 1] + mat[0, 1]) + (mat[0, 0] + mat[1, 0]) * (mat[0, 0] + mat[0, 1])) / (np.sum(mat) * np.sum(mat))
      ka = 1.0 * (acc - pe) / (1 - pe)

      sensitivity.append(sen)
      specificity.append(spe)
      precision.append(pre)
      accuracy.append(acc)
      F1_score.append(f1)
      kappa.append(ka)
      print('class{} - Sensitivity: {:.4f} - Specificity: {:.4f} - Accuracy: {:.4f} - F1-score: {:.4f} - Kappa: {:.4f}'.format(disease_abbr_6[i], sen, spe, acc, f1, ka)) 
   
   micro_avg_acc = correct / (correct + incorrect)
   macro_avg_acc = np.array(accuracy).mean()
   avg_f1 = np.array(F1_score).mean()
   avg_kappa = np.array(kappa).mean()

   print('Overall - Micro Accuracy: {:.4f}, Macro Accuracy: {:.4f}, F1_score: {:.4f}, Kappa: {:.4f}'.format(micro_avg_acc, macro_avg_acc, avg_f1, avg_kappa))
   print('Confusion Matrix: ')
   print(confusion_matrix)


calculate()




