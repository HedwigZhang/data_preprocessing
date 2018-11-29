"""
This file is to read Multi-view data for ModelNet40
One-view
"""
'''
 40 class = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
             'car', 'chair', 'cone, 'cup', 'curtain', 'desk', 'door', 'dresser', 
             'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
             'mantel', 'moniter', 'night_stand', 'person', 'piano', 'plant', 'radio',
             'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet',
             'tv_stand', 'vase', 'wardrobe', 'xbox'];
'''

import os
import cv2
import numpy as np
import pickle

# 数据集所在的地址
main_dir = "D:/dataset/ModelNet/modelnet40v1"
# 存储地址
save_dir = "D:/dataset/ModelNet"

# 创建空余的存储
traindata =[[]]
trainlabel = []
##trainname = []
testdata = [[]]
testlabel = []
#testname = []

# 第一类数据
i = 0
print("the numer of class", i)
# os.listdir(main_dir) 获取 main_dir 的所有文件
for main_name in os.listdir(main_dir):
    #print(main_name)  40类，每一类的文件夹
    # 将每一类添加到路径中，然后遍历子文件夹
    sub_dir = os.path.join(main_dir, main_name)
    # print(sub_dir)
    # 每一类下面的两个子文件夹： 'train' 和 'test'
    for sub_name in os.listdir(sub_dir):
        sub_sub_dir = os.path.join(sub_dir,sub_name)
        # 遍历训练子集下所有的图像，并且生成数据
        if sub_name == 'train':
        #print(sub_sub_dir)
            count_train = 0
            for sub_sub_name in os.listdir(sub_sub_dir):
                view = sub_sub_name.split('_')
                '''
                因为有些类名，经过分割后会分成四项，因此不同于其他的项
                '''
                len_view = len(view)
                if len_view == 3:
                    view2 = view[2]
                    if view2 == '001.jpg':
                        img_dir = os.path.join(sub_sub_dir,sub_sub_name)
                        print(img_dir)
                        img1 = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
                        #img1 = img1.reshape(1,50176)
                        img1 = img1.reshape([50176])
                        img1 = np.array(img1)
                        if traindata == [[]]:
                            traindata = [img1]
                        else:
                            traindata = np.concatenate((traindata, [img1]), axis = 0)
                        count_train = count_train + 1

                else:
                    view3 = view[3]
                    if view3 == '001.jpg':
                        img_dir = os.path.join(sub_sub_dir,sub_sub_name)
                        print(img_dir)
                        img1 = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
                        #img1 = img1.reshape(1,50176)
                        img1 = img1.reshape([50176])
                        img1 = np.array(img1)
                        if traindata == [[]]:
                            traindata = [img1]
                        else:
                            traindata = np.concatenate((traindata, [img1]), axis = 0)
                        count_train = count_train + 1
        # 遍历测试子集下所有的图像，并且生成数据
        if sub_name == 'test':
        #print(sub_sub_dir)
            count_test = 0
            for sub_sub_name in os.listdir(sub_sub_dir):
                view = sub_sub_name.split('_')
                '''
                因为有些类名，经过分割后会分成四项，因此不同于其他的项
                '''
                len_view = len(view)
                if len_view == 3:
                    view2 = view[2]
                    if view2 == '001.jpg':
                        img_dir = os.path.join(sub_sub_dir,sub_sub_name)
                        print("The dir of img", img_dir)
                        img1 = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
                        #img1 = img1.reshape(1,50176)
                        img1 = img1.reshape([50176])
                        img1 = np.array(img1)
                        if testdata == [[]]:
                            testdata = [img1]
                        else:
                            testdata = np.concatenate((testdata, [img1]), axis = 0)
                        count_test = count_test + 1
                else:
                    view3 = view[3]
                    if view3 == '001.jpg':
                        img_dir = os.path.join(sub_sub_dir,sub_sub_name)
                        print("The dir of img", img_dir)
                        img1 = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
                        #img1 = img1.reshape(1,50176)
                        img1 = img1.reshape([50176])
                        img1 = np.array(img1)
                        if testdata == [[]]:
                            testdata = [img1]
                        else:
                            testdata = np.concatenate((testdata, [img1]), axis = 0)
                        count_test = count_test + 1
        # 遍历测试子集下所有的图像，并且生成数据
    '''
    训练集标签
    '''
    trainlabels = np.zeros((count_train,))
    if trainlabel == []:
        trainlabel = trainlabels
    else:
        trainlabels[:] = i
        trainlabel = np.append(trainlabel, trainlabels)
    
    '''
    测试集标签
    '''
    testlabels = np.zeros((count_test,))
    if testlabel == []:
        testlabel = testlabels
    else:
        testlabels[:] = i
        testlabel = np.append(testlabel, testlabels)

    i = i + 1

    print("the numer of class", i)
# 数据类型转换，将数据转换成整型
trainlabel = trainlabel.astype(np.int32)
testlabel = testlabel.astype(np.int32)
#np.save(save_dir + 'train.npy', traindata)
# 创建字典来进行保存
train_dict = {'data': traindata, 'label': trainlabel}
test_dict = {'data': testdata, 'label': testlabel}

f_train = open('D:/dataset/ModelNet/data/train','wb')
f_test = open('D:/dataset/ModelNet/data/test','wb')

pickle.dump(train_dict, f_train)
pickle.dump(test_dict, f_test)