# cat_face_detection
调用tensorflow的object detection模块，利用自己的制作的数据集，基于已有的ssd_mobilenet_coco_v1模型训练猫脸识别的网络，能够识别猫脸和猫眼。
项目的最终应用是微信小程序中能够对猫脸进行美化、加装饰物等操作。
可能借鉴人脸特征点识别进而探讨猫脸特征点标注。

## 搭建步骤
1. 配置Tensorflow Object Detection API环境，参考官方文档[Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).  
2. 下载数据。Kaggle官网上有一个[Dogs vs. Cats](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)数据集，
其中的图片满足不同背景、光线情况，刚好可以做训练和测试集。  
3. 用LabelImg在2中下载的数据图片上标注猫脸和猫眼的位置，制作自己的数据集，生成.xml文件。可以参考[Raccoon](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9).  
4. 利用自己编写的脚本generate_tfrecord.sh将.xml文件先转化为.csv最终转化为tfRecord，可以进行tensorflow训练。  
5. 编写自己的训练配置，可以参考官方文档[Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)，主要包括label_map.pbtxt和ssd_mobilenet_v1_cat.config。  
6. 调用自己编写的脚本train_cat.sh进行训练，用tensorboard进行监视，记录checkpoint。  
7. 训练结束后将checkpoint文件导出为模型，参考官方文档[Exporting a trained model for inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md).
8. 测试测试！生成的图片在result文件里。

## 目录结构描述
```
├── README.md                   // help <br>
├── generate_tfrecord.sh        // 一次完成从xml文件转换为tfRecord的脚本，在本文件夹直接调用 <br>
├── xml_to_csv.py               // generate_tfrecord.sh调用的python程序 <br>
├── generate_tfrecord.py        // generate_tfrecord.sh调用的python程序 <br>
├── train_cat.sh                // 开启训练的脚本，需要在tensorflow/models/research下调用 <br>
├── test_cat.sh                 // 开启测试的脚本，需要在tensorflow/models/research下调用 <br>
├── tensorflow.sh               // 开启tensorboard的脚本，任意位置调用 <br>
├── my_object_detection.py      // 测试训练得到模型，需要在tensorflow/models/research/object_detection下调用 <br>
├── annotations                 // 注释，包含手动标注的数据集，共100张 <br>
│   ├── train                   // 手动标注训练数据集，xml, 80张 <br>
│   └── test                    // 手动标注测试数据集，xml, 20张 <br>
├── data                        // 由annotations生成的csv和tfRecord文件 <br>
│   ├── train_labels.csv        // 训练集csv文件 <br>
│   ├── test_labels.csv         // 测试集csv文件 <br>
│   ├── train.record            // 训练集tfRecord文件 <br>
│   └── teset.record            // 测试集tfRecord文件 <br>
├── images                      // 200张来自原数据集的图片 <br>
├── results                     // 100张训练后模型的测试结果 <br>
└── training                    // 训练模型有关文件 <br>
    ├── label_map.pbtxt         // label map，2个label <br>
    ├── ssd_mobilnet_v1_cat.config // 训练配置文件 <br>
    ├── ssd_mobilenet_v1_coco_2017_11_17.tar.gz // 基于的已有模型 <br>
    ├── ssd_mobilenet_v1_coco_2017_11_17        // 已有模型的解压文件 <br>
    ├── trainlog               // 训练checkpoint记录和tensorboard event记录, TRAIN_DIR <br>
    ├── evallog                // 测试tensorboard event记录, EVAL_DIR <br>
    └── exported_model         // 从trainlog中的checkpoint中导出来的模型，EXPORT_DIR <br>
```
