# Introduction

使用了 [tensorflow-serving-yolov3](https://github.com/Byronnar/tensorflow-serving-yolov3) 的算法，原版的 yolo 算法训练的过程对 GPU 要求较高，因此使用了 tensorflow 版本的 yolo 算法。

在原版的 tensorflow-serving-yolov3 上主要修改了：

- `./core/config.py`

  ~~~
  __C.YOLO.CLASSES => class_names
  __C.TRAIN.ANNOT_PATH => train_labels
  __C.TEST.ANNOT_PATH => test_labels
  __C.TRAIN.BATCH_SIZE => banch_size 根据显存大小调整
  ~~~

- 增加了 VisDrone 数据集，位置在 `./VisDrone2018-tf-yolo/`

  训练好的模型在 `./yolov3_visdrone.pb`

目前检测 VisDrone 数据集的正确率已经比较高了

![mAP](./mAP/mAP.png)

~~目前只做好了检查，还没有做计数的功能!~~

目前已经做好了计数功能。

# Part0. 资源下载

[yolov3_visdrone.pb(训练好的模型百度网盘链接)](https://pan.baidu.com/s/185IsU0JwSDpu4_rM1Nd-Zw) 提取码: 8jqg

[VisDrone2018-tf-yolo(筛选过的数据集百度网盘链接)](https://pan.baidu.com/s/1BdEpondAfU2qovHjXukXQQ) 提取码: ux34

# Part1. Demo展示

1. 进入文件夹，安装需要的软件

   ~~~
   pip3 install -r requirements.txt
   ~~~

   **这里安装的是GPU版的Tensorflow，CPU版的Tensorflow也是可以用的，只是运行速度慢一些**

2. 模型已经训练好，直接运行

   ~~~bash
   ./video_demo_new.sh
   ~~~

   待检测的视频是 `./docs/images/1.mp4`，如需修改待检测视频需要修改 `video_demo.py` 第14行的 `video_path`，代码中有 resize 的过程，因此视频的尺寸没有要求
   
   **运行的时候可能还缺少一些库，按照报错原因安装相应的库**
   
3. 检测的结果在 `./output/demo.mp4` `./output/counting.mp4` 

# Part2. 程序说明

## 1. 目录结构

~~~
.
|____readme_images
|____tools                                  # 为车流量统计写的一些程序
| |____image_process.py
| |____speed_prediction.py
| |____save_image.py
| |____vehicle_counting.py
| |____iou_tracker.py                       # iou tracker
| |____trackers_to_perframe.py
| |____vis_tracker.py
|____yolov3_visdrone.pb                     # 车辆模型
|____checkpoint                             # 训练自定义数据集检查点存放的位置
|____core                                   # tensorflow-yolov3 核心代码
| |____config.py                            # 配置文件
| |____...
|____vehicle_counting.py                    # 车流量统计的主程序 2
|____requirements.txt                       # 项目所需的库
|____pickle_file_path
|____image_demo.py
|____video_demo_new.py                      # 车流量统计的主程序 1
|____VisDrone2018-tf-yolo                   # 车辆数据集
|____serving-yolov3
|____image_demo_Chinese.py
|____output
| |____test_output
| | |____tmp.pk                             # 两端程序的中间数据
| | |____...
| |____demo.mp4                             # 车流量统计的主程序 1(video_demo_new.py) 输出的结果
| |____counting.mp4                         # 车流量统计的主程序 2(vehicle_counting.py) 输出的结果
| |____...
|____save_model.py
|____tensorboard.sh                         # 可视化训练模型
|____README.md
|____train.py
|____video_demo.py
|____evaluate.py
|____convert_weight.py
|____video_demo_new.sh                      # 对两段主程序的封装
|____...
~~~

## 2. 程序说明

当执行 `./video_demo_new.sh` 时，会执行两段程序：

1. 首先执行 `video_demo_new.py`，这段程序包含了两个过程：

   1. 先试用训练好模型 `./yolov3_visdrone.pb` 对视频 `./docs/images/1.mp4` 进行检测，标出每帧图像中车辆的位置，这个过程同时会将检查的结果输出到 `./output/demo.mp4` 中；
   2. 接着使用 [iou-tracker](https://github.com/bochinski/iou-tracker) 算法进行多目标追踪，即在原先检测的结果上，标定出**每辆车的行驶轨迹**，这个过程的结果保存在 `./output/test_output/tmp.pk` 中；

   **在 `video_demo_new.py` 中有一段 `showVideo_flag = False`，这个表示在检测时实时显示检测结果，如果想观看实时结果可将其改为 `showVideo_flag = True`，由于这个过程是在 GPU 上进行的，显示可能会造成显存不够，程序奔溃，可以在程序运行完成后查看 `./output/demo.mp4`。**

2. 第二段程序 `vehicle_counting.py`，统计车流量，即使用之前检测出的车辆轨迹数据，当有车辆轨迹经过路口画的线时，进行统计。

   - **在 `video_demo_new.py` 中有一段 `showVideo_flag = False`，这个表示在统计时实时显示检测结果，建议将其改为 `showVideo_flag = True`，可以观看实时结果，也可以在程序运行完成后查看 `output/counting.mp4`。**
   
   - 路口划线位置在 `tools/trackers_to_perframe.py` 里修改：
   
     ~~~python
     # 左边路口划线位置
     LEFT_INTERSECTION_ROI_POSITION = 400
     LEFT_INTERSECTION_ROI_START = 300
     LEFT_INTERSECTION_ROI_END = 550
     # 右边路口划线位置
     RIGHT_INTERSECTION_ROI_POSITION = 1000
     RIGHT_INTERSECTION_ROI_START = 0
     RIGHT_INTERSECTION_ROI_END = 280
     # 底部路口划线位置
     BOTTOM_INTERSECTION_ROI_POSITION = 500
     BOTTOM_INTERSECTION_ROI_START = 500
     BOTTOM_INTERSECTION_ROI_END = 900
     ~~~

# Part3. 训练自定义数据集

参考 [Train your own dataset](https://github.com/YunYang1994/tensorflow-yolov3#part-2-train-your-own-dataset)

关于文件夹中的 VisDrone 数据集是我从原版的 [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) 数据集中选出来的有车辆信息的一部分，如果需要重新训练需要执行以下脚本，重新生成 label 文件

~~~
./VisDrone2018-tf-yolo/scripts/visdrone2tfyolo.sh
~~~

# TODO

- [x] ~~在检测前对图像进行预处理，调整对比度和亮度，参考 [使用TensorFlow对象检测API进行实时目标检测]([https://cjh.zone/2019/01/19/%E4%BD%BF%E7%94%A8TensorFlow%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8BAPI%E8%BF%9B%E8%A1%8C%E5%AE%9E%E6%97%B6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/#%E5%AE%9E%E6%97%B6%E6%A3%80%E6%B5%8B](https://cjh.zone/2019/01/19/使用TensorFlow对象检测API进行实时目标检测/#实时检测))目前在光照条件不好的情况下检测的准确率会下降~~

- [ ] 使用Python多线程，提高检测过程的速度，参考 [使用TensorFlow对象检测API进行实时目标检测]([https://cjh.zone/2019/01/19/%E4%BD%BF%E7%94%A8TensorFlow%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8BAPI%E8%BF%9B%E8%A1%8C%E5%AE%9E%E6%97%B6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/#%E5%AE%9E%E6%97%B6%E6%A3%80%E6%B5%8B](https://cjh.zone/2019/01/19/使用TensorFlow对象检测API进行实时目标检测/#实时检测))

  ![img](https://cjh.zone/2019/01/19/%E4%BD%BF%E7%94%A8TensorFlow%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8BAPI%E8%BF%9B%E8%A1%8C%E5%AE%9E%E6%97%B6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/1547808115848.png)

- [x] ~~对车流量计数，参考 [vehicle_counting_tensorflow](https://github.com/ahmetozlu/vehicle_counting_tensorflow)~~

# Reference

[tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

[tensorflow-serving-yolov3](https://github.com/Byronnar/tensorflow-serving-yolov3)

[iou-tracker](https://github.com/bochinski/iou-tracker)

