#! /usr/bin/env python
# coding=utf-8


import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import pickle
# 进度条
from tqdm import tqdm

import tools.save_image as save_image

from tools.iou_tracker import save_mod, track_viou_video, save_to_csv

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_visdrone.pb"
video_path      = "./docs/images/1.mp4"
# video_path      = 0
num_classes     = 12
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

output_path = './output/demo.mp4'

showVideo_flag = False
writeVideo_flag = True

annotation_path = 'output/test_output/tracker.txt'
pickle_file_path = 'output/test_output/tmp.pk'

processing_flag = True


with tf.Session(graph=graph) as sess:
    if writeVideo_flag:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")

        # 总帧数
        total_frame_counter = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print('总帧数:', total_frame_counter)

        video_FourCC = cv2.VideoWriter_fourcc(*'MP4V')
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        isOutput = True if output_path != "" else False
        if isOutput:
            #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
            list_file = open('detection.txt', 'w')
            frame_index = -1

    # 创建进度条
    if not showVideo_flag:
        pbar = tqdm(total=total_frame_counter)

    while True:
        return_value, frame = vid.read()
        if return_value != True:
            break
        if return_value:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            # print('image:',image)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})
        pred_time = time.time()

        # print('time:',pred_time-prev_time)

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.45)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        # save images
        # save_image.save_image(annotation_file, frame, vid.get(1), bboxes)

        image = utils.draw_bbox(frame, bboxes)
        # image = utils.draw_bbox(frame, bboxes, vid.get(1))
        # 保存为 iou_tracker 格式
        detections = save_mod(bboxes, 0.6)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: " + str(format(round(1000 * exec_time, 2), '^5.2f')) + " ms, FPS: " + str(format(round((1000 / (1000 * exec_time)), 1), '^5.2f')) + '  '
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if writeVideo_flag:
            # save a frame
            out.write(result)

        if showVideo_flag:
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            pbar.update(1)
            processing_info = info
            # print('\r',processing_info.ljust(50),end='',flush=True)
            print('\r', processing_info, end='', flush=True)

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()   

# 多目标追踪
trackers = track_viou_video(video_path ,detections , 0.5, 0.6, 0.1, 23, 16, 'MEDIANFLOW', 1.0)

# 保存 trackers
with open(pickle_file_path, 'wb') as pk_f:
    pickle.dump(trackers, pk_f)
    print('=> saved trackers to pk file.')


