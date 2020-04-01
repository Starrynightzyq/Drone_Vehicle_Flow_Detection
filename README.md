# Introduction

使用了 [tensorflow-yolov3](https://github.com/Byronnar/tensorflow-serving-yolov3) 的算法，原版的 yolo 算法训练的过程对 GPU 要求较高，因此使用了 tensorflow 版本的 yolo 算法。

在原版的 tensorflow-yolov3 上主要修改了：

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

![mAP](mAP/mAP.png)

**目前只做好了检查，还没有做计数的功能！**

# Part1. Demo展示

1. 进入文件夹，安装需要的软件

   ~~~
   pip3 install -r requirements.txt
   ~~~

   **这里安装的是GPU版的Tensorflow，CPU版的Tensorflow也是可以用的，只是运行速度慢一些**

2. 模型已经训练好，直接运行

   ~~~
   python3 video_demo.py
   ~~~

   待检测的视频是 `./docs/images/1.mp4`，如需修改待检测视频需要修改 `video_demo.py` 第14行的 `video_path`，代码中有 resize 的过程，因此视频的尺寸没有要求
   
   **运行的时候可能还缺少一些库，按照报错原因安装相应的库**
   
3. 检测的结果在 `./output/demo.mp4`

# Part2. 训练自定义数据集

参考 [Train your own dataset](https://github.com/YunYang1994/tensorflow-yolov3#part-2-train-your-own-dataset)

关于文件夹中的 VisDrone 数据集是我从原版的 [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) 数据集中选出来的有车辆信息的一部分，如果需要重新训练需要执行以下脚本，重新生成 label 文件

~~~
./VisDrone2018-tf-yolo/scripts/visdrone2tfyolo.sh
~~~

# TODO

- [ ] 在检测前对图像进行预处理，调整对比度和亮度，参考 [使用TensorFlow对象检测API进行实时目标检测]([https://cjh.zone/2019/01/19/%E4%BD%BF%E7%94%A8TensorFlow%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8BAPI%E8%BF%9B%E8%A1%8C%E5%AE%9E%E6%97%B6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/#%E5%AE%9E%E6%97%B6%E6%A3%80%E6%B5%8B](https://cjh.zone/2019/01/19/使用TensorFlow对象检测API进行实时目标检测/#实时检测))

  目前在光照条件不好的情况下检测的准确率会下降

- [ ] 使用Python多线程，提高检测过程的速度，参考 [使用TensorFlow对象检测API进行实时目标检测]([https://cjh.zone/2019/01/19/%E4%BD%BF%E7%94%A8TensorFlow%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8BAPI%E8%BF%9B%E8%A1%8C%E5%AE%9E%E6%97%B6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/#%E5%AE%9E%E6%97%B6%E6%A3%80%E6%B5%8B](https://cjh.zone/2019/01/19/使用TensorFlow对象检测API进行实时目标检测/#实时检测))

  ![img](https://cjh.zone/2019/01/19/%E4%BD%BF%E7%94%A8TensorFlow%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8BAPI%E8%BF%9B%E8%A1%8C%E5%AE%9E%E6%97%B6%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/1547808115848.png)

- [ ] 对车流量计数，参考 [vehicle_counting_tensorflow](https://github.com/ahmetozlu/vehicle_counting_tensorflow)

# Reference

[tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

[tensorflow-serving-yolov3](https://github.com/Byronnar/tensorflow-serving-yolov3)
