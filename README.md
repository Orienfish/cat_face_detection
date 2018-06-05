# cat_face_detection
调用tensorflow的object detection模块，利用自己的制作的数据集，基于已有的ssd_mobilenet_coco_v1模型训练猫脸识别的网络，能够识别猫脸和猫眼。
项目的最终应用是微信小程序中能够对猫脸进行美化、加装饰物等操作。
可能借鉴人脸特征点识别进而探讨猫脸特征点标注。

## 搭建步骤
1. 配置Tensorflow Object Detection API环境，参考官方文档[Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).  
2. 下载数据。Kaggle官网上有一个[Dogs vs. Cats](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)数据集，
其中的图片满足不同背景、光线情况，刚好可以做训练和测试集。  
3. 用LabelImg在2中下载的数据图片上标注猫脸和猫眼的位置，制作自己的数据集，生成.xml文件。可以参考[Raccoon]
(https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9).  
4. 利用自己编写的脚本generate_tfrecord.sh将.xml文件先转化为.csv最终转化为tfRecord，可以进行tensorflow训练。  
5. 编写自己的训练配置，可以参考官方文档[Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)，
主要包括label_map.pbtxt和ssd_mobilenet_v1_cat.config。  
6. 调用自己编写的脚本train_cat.sh进行训练，用tensorboard进行监视，记录checkpoint。  
7. 训练结束后将checkpoint文件导出为模型，参考官方文档[Exporting a trained model for inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md).
8. 测试测试！生成的图片在result文件里。

