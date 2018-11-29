# data_preprocessing
The pre-processing stage of some data sets. Converting data from pictures to .py data

ImageNet_train.py
是对ILSVRC2012中的训练集进行处理。
需要提前下载数据集:
1.官网下载：http://image-net.org/download-images
2：https://blog.csdn.net/xingchengmeng/article/details/58135148

tensorflow上也提供了对数据进行处理，转换成tfrecord的方法，详细如下：https://blog.csdn.net/Gavin__Zhou/article/details/80242998

我对于tfrecord的理解不深，因此自己写代码提供了如下处理方案。
对   ILSVRC2012_img_train.tar  解压缩，然后一次遍历每一个子 压缩包。将图像重新调整为统一的227*227*3的尺寸。


DataForModelNet40.py

对ModelNet40数据集进行处理，其特点子文件夹的深度为3，徐遍历所有的子文件夹。
