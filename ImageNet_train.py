import tarfile
import numpy as np
import os
#import cv2
import pickle
from PIL import Image, TarIO

data_dir = 'D:/Code/pythonFile/models/research/slim/ImageNet-ori/ILSVRC2012_img_train'

# Storage training data
traindata =[[]]
trainlabel = []
trainname = []
# design this for each class which will be used in contructuring val data
trainname_class = []
# 初始类名，重新构建类的标签，1000类，因此为0-999
trainlabel_label = np.arange(1000)
# os.listdir(data_dir) 获取 data_dir 的所有文件
j = 0
for main_name in os.listdir(data_dir):
    # 将tar文件加入到地址中，进而遍历，获取类名，每一个类似于 ‘n01440764.tar’,
    # 进入训练集，里面还会有1000个压缩包，分别代表一类，其中包含很多图像
    sub_dir = os.path.join(data_dir, main_name)
    print("The dir of sub class", sub_dir)
    class_name = main_name.split('.')[0]
    trainname_class.append(class_name)
    file = tarfile.open(sub_dir, "r")
    file_list = []
    # 遍历每一个 tar 文件， 进而学习特征
    for i in file.getmembers():
        # 获取文件名，每一个类似于 ‘n01440764.tar’
        file_name = i.name
        print("name of pictures ", file_name)
        file_list.append(file_name)
        fp = TarIO.TarIO(sub_dir, file_name)
        # 打开图片
        im = Image.open(fp)
        # 对所有的图片进行裁剪，重新标定尺寸
        img = im.resize((227, 227),Image.ANTIALIAS)
        #从Image类转化为 numpy array, 事实上此时的图片已经被转换成数组，227*227*3。
        img = np.asarray(img)
        if img.size == 227*227:
            img = np.concatenate((img, img, img), axis = 0)

        img = np.array(img)
        # reshape 图片，方便放入dicts
        img = img.reshape([227*227*3])
        if traindata == [[]]:
            traindata = [img]
        else:
            traindata = np.concatenate((traindata, [img]), axis = 0)

        im.close()

    len_class = len(file_list)
    trainlabels = np.zeros((len_class,))
    if trainlabel == []:
        trainlabel = trainlabels
    else:
        trainlabels[:] = j
        trainlabel = np.append(trainlabel, trainlabels)

    j = j + 1

# 数据类型转换，将数据转换成整型
trainlabel = trainlabel.astype(np.int32)
# 创建字典来进行保存
train_dict = {'data': traindata, 'label': trainlabel}
label_test_dict = {'class': trainname_class, 'label': trainlabel_label}

f_train = open('D:/dataset/ImageNet/train','wb')
label_test = open('D:/dataset/ImageNet/label_test','wb')

pickle.dump(train_dict, f_train)
pickle.dump(label_test_dict, label_test)