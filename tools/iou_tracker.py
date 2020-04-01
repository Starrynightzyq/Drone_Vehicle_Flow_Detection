import math
import numpy as np
import sys
import os
import csv
from tools.vis_tracker import VisTracker
import cv2
from lapsolver import solve_dense
from tqdm import tqdm
from time import time

mod_data = []

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def giou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    # 计算交集面积
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    # 计算并集面积
    size_union = size_1 + size_2 - size_intersection
    # 计算 iou
    iou_value = size_intersection / size_union

    return size_intersection / size_union


# 去除同一帧图像中交叠过大的检测框
def nms(boxes, scores, overlapThresh, classes=None):
    """
    perform non-maximum suppression. based on Malisiewicz et al.
    Args:
        boxes (numpy.ndarray): boxes to process
        scores (numpy.ndarray): corresponding scores for each box
        overlapThresh (float): overlap threshold for boxes to merge
        classes (numpy.ndarray, optional): class ids for each box.

    Returns:
        (tuple): tuple containing:

        boxes (list): nms boxes
        scores (list): nms scores
        classes (list, optional): nms classes if specified
    """
    # # if there are no boxes, return an empty list
    # if len(boxes) == 0:
    #     return [], [], [] if classes else [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #score = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]


def associate(tracks, detections, sigma_iou):
    """ perform association between tracks and detections in a frame.
    Args:
        tracks (list): input tracks
        detections (list): input detections
        sigma_iou (float): minimum intersection-over-union of a valid association

    Returns:
        (tuple): tuple containing:

        track_ids (numpy.array): 1D array with indexes of the tracks
        det_ids (numpy.array): 1D array of the associated indexes of the detections
    """
    costs = np.empty(shape=(len(tracks), len(detections)), dtype=np.float32)
    for row, track in enumerate(tracks):
        for col, detection in enumerate(detections):
            costs[row, col] = 1 - iou(track['bboxes'][-1], detection['bbox'])

    np.nan_to_num(costs)
    costs[costs > 1 - sigma_iou] = np.nan
    track_ids, det_ids = solve_dense(costs)
    return track_ids, det_ids


def track_viou_video(video_path, detections, sigma_l, sigma_h, sigma_iou, t_min, ttl, tracker_type, keep_upper_height_ratio):
    """ V-IOU Tracker.
    See "Extending IOU Based Multi-Object Tracking by Visual Information by E. Bochinski, T. Senst, T. Sikora" for
    more information.

    Args:
         frames_path (str): path to ALL frames.
                            string must contain a placeholder like {:07d} to be replaced with the frame numbers.
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.
         ttl (float): maximum number of frames to perform visual tracking.
                      this can fill 'gaps' of up to 2*ttl frames (ttl times forward and backward).
         tracker_type (str): name of the visual tracker to use. see VisTracker for more details.
         keep_upper_height_ratio (float): float between 0.0 and 1.0 that determines the ratio of height of the object
                                          to track to the total height of the object used for visual tracking.

    Returns:
        list: list of tracks.
    """
    if tracker_type == 'NONE':
        assert ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"

    tracks_active = []
    tracks_extendable = []
    tracks_finished = []
    frame_buffer = []

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    for frame_num, detections_frame in enumerate(tqdm(detections), start=1):
        # load frame and put into buffer
        # frame_path = frames_path.format(frame_num)
        # frame = cv2.imread(frame_path)
        
        return_value, frame = vid.read()
        if return_value != True:
            break
        if return_value:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(frame)
            # print('image:',image)
            pass
        else:
            raise ValueError("No image!")

        
        assert frame is not None, "could not read '{}'".format(frame_path)
        frame_buffer.append(frame)
        if len(frame_buffer) > ttl + 1:
            frame_buffer.pop(0)

        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        track_ids, det_ids = associate(tracks_active, dets, sigma_iou)
        updated_tracks = []
        for track_id, det_id in zip(track_ids, det_ids):
            tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
            tracks_active[track_id]['max_score'] = max(tracks_active[track_id]['max_score'], dets[det_id]['score'])
            tracks_active[track_id]['classes'].append(dets[det_id]['class'])
            tracks_active[track_id]['det_counter'] += 1

            if tracks_active[track_id]['ttl'] != ttl:
                # reset visual tracker if active
                tracks_active[track_id]['ttl'] = ttl
                tracks_active[track_id]['visual_tracker'] = None

            updated_tracks.append(tracks_active[track_id])

        tracks_not_updated = [tracks_active[idx] for idx in set(range(len(tracks_active))).difference(set(track_ids))]

        for track in tracks_not_updated:
            if track['ttl'] > 0:
                if track['ttl'] == ttl:
                    # init visual tracker
                    track['visual_tracker'] = VisTracker(tracker_type, track['bboxes'][-1], frame_buffer[-2],
                                                         keep_upper_height_ratio)
                # viou forward update
                ok, bbox = track['visual_tracker'].update(frame)

                if not ok:
                    # visual update failed, track can still be extended
                    tracks_extendable.append(track)
                    continue

                track['ttl'] -= 1
                track['bboxes'].append(bbox)
                updated_tracks.append(track)
            else:
                tracks_extendable.append(track)

        # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
        # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
        tracks_extendable_updated = []
        for track in tracks_extendable:
            if track['start_frame'] + len(track['bboxes']) + ttl - track['ttl'] >= frame_num:
                tracks_extendable_updated.append(track)
            elif track['max_score'] >= sigma_h and track['det_counter'] >= t_min:
                tracks_finished.append(track)
        tracks_extendable = tracks_extendable_updated

        new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
        dets_for_new = []

        for det in new_dets:
            finished = False
            # go backwards and track visually
            boxes = []
            vis_tracker = VisTracker(tracker_type, det['bbox'], frame, keep_upper_height_ratio)

            for f in reversed(frame_buffer[:-1]):
                ok, bbox = vis_tracker.update(f)
                if not ok:
                    # can not go further back as the visual tracker failed
                    break
                boxes.append(bbox)

                # sorting is not really necessary but helps to avoid different behaviour for different orderings
                # preferring longer tracks for extension seems intuitive, LAP solving might be better
                for track in sorted(tracks_extendable, key=lambda x: len(x['bboxes']), reverse=True):

                    offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - frame_num
                    # association not optimal (LAP solving might be better)
                    # association is performed at the same frame, not adjacent ones
                    if 1 <= offset <= ttl - track['ttl'] and iou(track['bboxes'][-offset], bbox) >= sigma_iou:
                        if offset > 1:
                            # remove existing visually tracked boxes behind the matching frame
                            track['bboxes'] = track['bboxes'][:-offset+1]
                        track['bboxes'] += list(reversed(boxes))[1:]
                        track['bboxes'].append(det['bbox'])
                        track['max_score'] = max(track['max_score'], det['score'])
                        track['classes'].append(det['class'])
                        track['ttl'] = ttl
                        track['visual_tracker'] = None

                        tracks_extendable.remove(track)
                        if track in tracks_finished:
                            del tracks_finished[tracks_finished.index(track)]
                        updated_tracks.append(track)

                        finished = True
                        break
                if finished:
                    break
            if not finished:
                dets_for_new.append(det)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num, 'ttl': ttl,
                       'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None} for det in dets_for_new]
        tracks_active = []
        for track in updated_tracks + new_tracks:
            if track['ttl'] == 0:
                tracks_extendable.append(track)
            else:
                tracks_active.append(track)

    # finish all remaining active and extendable tracks
    tracks_finished = tracks_finished + \
                      [track for track in tracks_active + tracks_extendable
                       if track['max_score'] >= sigma_h and track['det_counter'] >= t_min]

    # remove last visually tracked frames and compute the track classes
    for track in tracks_finished:
        if ttl != track['ttl']:
            track['bboxes'] = track['bboxes'][:-(ttl - track['ttl'])]
        track['class'] = max(set(track['classes']), key=track['classes'].count)

        del track['visual_tracker']

    
    # debug
    # print(data)
    f = open('debug.txt', 'w')
    f.write(str(tracks_finished))
    f.close()


    return tracks_finished


# visdrone_classes = {'car': 4, 'bus': 9, 'truck': 6, 'pedestrian': 1, 'van': 5}
visdrone_classes = {'ignored-regions':0,'pedestrian':1,'people':2,'bicycle':3,'car':4,'van':5,'truck':6,'tricycle':7,'awning-tricycle':8,'bus':9,'motor':10,'others':11}
visdrone_class_name = ['ignored-regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']


# save data to mod
def save_mod(bboxes_raw, nms_overlap_thresh=0.6):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).

    Args:
        detections (str, numpy.ndarray): path to csv file containing the detections or numpy array containing them.
        nms_overlap_thresh (float, optional): perform non-maximum suppression on the input detections with this thrshold.
                                              no nms is performed if this parameter is not specified.
        with_classes (bool, optional): indicates if the detections have classes or not. set to false for motchallange.
        nms_per_class (bool, optional): perform non-maximum suppression for each class separately

    Returns:
        list: list containing the detections for each frame.
    """

    global mod_data

    bboxes_raw = np.array(bboxes_raw, dtype=np.float32)

    dets = []
    bboxes = np.array(bboxes_raw[:, 0:4], dtype=np.int32) # x1, y1, x2, y2
    # bboxes = bboxes_raw[:, 0:4] # x1, y1, x2, y2
    scores = bboxes_raw[:, 4]
    # classes = visdrone_class_name[int(bboxes_raw[:, 5])]
    classes = np.array(bboxes_raw[:, 5], dtype=np.int32)

    bboxes, scores, classes = nms(bboxes, scores, nms_overlap_thresh, classes)

    # for target_index, bbox_raw in enumerate(bboxes_raw):
    #     bbox = np.array(bbox_raw[:4], dtype=np.int32) # x1, y1, x2, y2
    #     # bbox -= 1  # correct 1,1 matlab offset
    #     score_value = round(bbox_raw[4],2) # 取两位小数
    #     class_name = visdrone_class_name[int(bbox_raw[5])]

    for bb, s, c in zip(bboxes, scores, classes):
        dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': 1.0, 'class': visdrone_class_name[c]})
        # dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': visdrone_class_name[c]})

    mod_data.append(dets)

    # print(str(mod_data))

    return mod_data


def save_to_csv(out_path, tracks, fmt='motchallenge'):
    """
    Saves tracks to a CSV file.

    Args:
        out_path (str): path to output csv file.
        tracks (list): list of tracks to store.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as ofile:
        if fmt == 'motchallenge':
            field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']
        elif fmt == 'visdrone':
            field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'object_category', 'truncation', 'occlusion']
        else:
            raise ValueError("unknown format type '{}'".format(fmt))

        odict = csv.DictWriter(ofile, field_names)
        id_ = 1
        for track in tracks:
            for i, bbox in enumerate(track['bboxes']):
                row = {'id': id_,
                       'frame': track['start_frame'] + i,
                       'x': bbox[0]+1,
                       'y': bbox[1]+1,
                       'w': bbox[2] - bbox[0],
                       'h': bbox[3] - bbox[1],
                       'score': track['max_score']}
                if fmt == 'motchallenge':
                    row['wx'] = -1
                    row['wy'] = -1
                    row['wz'] = -1
                elif fmt == 'visdrone':
                    row['object_category'] = visdrone_classes[track['class']]
                    row['truncation'] = -1
                    row['occlusion'] = -1
                else:
                    raise ValueError("unknown format type '{}'".format(fmt))

                odict.writerow(row)
            id_ += 1
