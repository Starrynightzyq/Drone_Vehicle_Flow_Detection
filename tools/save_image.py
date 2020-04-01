import pandas as pd
import numpy as np
import cv2

sequences_dir = '/Users/zhouyugan/MyDocuments/vehicle_flow/tensorflow-serving-yolov3/output/test_output/sequences/'

head_name = ['frame_index','target_id','bbox_left','bbox_top','bbox_width','bbox_height','score','object_category','truncation','occlusion']
class_name = ('ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others')

def save_image(file, img, img_index, bboxes):

    # file = open(annotation_path, 'w')

    for target_index, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        # score = bbox[4]
        score = 1.0
        class_ind = int(bbox[5])
        # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        truncation = -1
        occlusion = -1

        file_content = str(int(img_index)) + ',' \
            + str(target_index) + ',' \
            + str(coor[0]) + ',' \
            + str(coor[1]) + ',' \
            + str(coor[2] - coor[0]) + ',' \
            + str(coor[3] - coor[1]) + ',' \
            + str(round(score,2)) + ',' \
            + str(class_ind) + ',' \
            + '-1' + ',' + '-1' + '\n'

        file.write(file_content)

        img_name = sequences_dir + str(int(img_index)).zfill(7)+'.jpg'
        cv2.imwrite(img_name, img)