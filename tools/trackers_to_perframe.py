import cv2
import pdb

left_vehicle_counter = 0
right_vehicle_counter = 0
bottom_vehicle_counter = 0

LEFT_INTERSECTION_ROI_POSITION = 400
LEFT_INTERSECTION_ROI_START = 300
LEFT_INTERSECTION_ROI_END = 550

RIGHT_INTERSECTION_ROI_POSITION = 1000
RIGHT_INTERSECTION_ROI_START = 0
RIGHT_INTERSECTION_ROI_END = 280

BOTTOM_INTERSECTION_ROI_POSITION = 500
BOTTOM_INTERSECTION_ROI_START = 500
BOTTOM_INTERSECTION_ROI_END = 900

bbox_color = {'ignored-regions':(0xFF,0x66,0x00),
    'pedestrian':(0xCC,0x66,0x00),
    'people':(0x99,0x66,0x00),
    'bicycle':(0x66,0x66,0x00),
    'car':(0x33,0xFF,0x00),
    'van':(0x00,0x66,0x00),
    'truck':(0xFF,0xFF,0x00),
    'tricycle':(0xCC,0xFF,0x00),
    'awning-tricycle':(0x99,0xFF,0x00),
    'bus':(0x66,0xFF,0x00),
    'motor':(0x33,0x66,0x00),
    'others':(0x00,0xFF,0x00)}

def trackers_to_perframe(trackers):
    pass

def draw_bbox_with_counting(image, image_index, trackers, show_box=True, show_label=True):

    global left_vehicle_counter
    global right_vehicle_counter
    global bottom_vehicle_counter

    for object_id, object_info in enumerate(trackers):
        # print(object_id, object_info)
        if image_index >= object_info['start_frame'] and \
            image_index < (object_info['start_frame']+len(object_info['bboxes'])):

            object_info_index = int(image_index - object_info['start_frame'])

            # bbox = object_info['bboxes'][object_info_index]
            # print(bbox)
            # pdb.set_trace()

            bbox = object_info['bboxes'][object_info_index]

            c1 = (int(bbox[0]), int(bbox[1]))
            c2 = (int(bbox[2]), int(bbox[3]))

            if show_box:  
                cv2.rectangle(image, c1, c2, bbox_color[object_info['class']], 2)
            if show_label:
                bbox_mess = '%s: %.2f' % (object_info['class'], object_info['max_score'])
                cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            # 判断是否是机动车
            if object_info['class'] in ('car', 'van', 'truck', 'bus'):       
                if object_info_index > 0:
                    bbox_last = object_info['bboxes'][object_info_index-1]
                    center_now = (((bbox[0])+(bbox[2]))/2, ((bbox[1])+(bbox[3]))/2)
                    center_last = (((bbox_last[0])+(bbox_last[2]))/2, ((bbox_last[1])+(bbox_last[3]))/2)

                    left_vehicle_is_online = counting_horizontal(center_now, center_last, \
                        LEFT_INTERSECTION_ROI_POSITION, \
                        LEFT_INTERSECTION_ROI_START, \
                        LEFT_INTERSECTION_ROI_END)

                    right_vehicle_is_online = counting_horizontal(center_now, center_last, \
                        RIGHT_INTERSECTION_ROI_POSITION, \
                        RIGHT_INTERSECTION_ROI_START, \
                        RIGHT_INTERSECTION_ROI_END)

                    bottom_vehicle_is_online = counting_vertical(center_now, center_last, \
                        BOTTOM_INTERSECTION_ROI_POSITION, \
                        BOTTOM_INTERSECTION_ROI_START, \
                        BOTTOM_INTERSECTION_ROI_END)

                    if left_vehicle_is_online:
                        left_vehicle_counter += 1
                    if right_vehicle_is_online:
                        right_vehicle_counter += 1
                    if bottom_vehicle_is_online:
                        bottom_vehicle_counter += 1

    
    cv2.line(image, \
        (BOTTOM_INTERSECTION_ROI_START, BOTTOM_INTERSECTION_ROI_POSITION), \
        (BOTTOM_INTERSECTION_ROI_END, BOTTOM_INTERSECTION_ROI_POSITION), \
        (0, 0, 0xFF), 5)
    cv2.line(image, \
        (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_START), \
        (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_END), \
        (0, 0, 0xFF), 5)
    cv2.line(image, \
        (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_START), \
        (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_END), \
        (0, 0, 0xFF), 5)

    left_info = str(left_vehicle_counter)
    right_info = str(right_vehicle_counter)
    bottom_info = str(bottom_vehicle_counter)
    cv2.putText(image, text=left_info, 
                org=(LEFT_INTERSECTION_ROI_POSITION+10, LEFT_INTERSECTION_ROI_START+20), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)
    cv2.putText(image, text=right_info, 
                org=(RIGHT_INTERSECTION_ROI_POSITION-60, RIGHT_INTERSECTION_ROI_END-10), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)
    cv2.putText(image, text=bottom_info, 
                org=(BOTTOM_INTERSECTION_ROI_START+10, BOTTOM_INTERSECTION_ROI_POSITION-20), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

    return image

def counting_horizontal(center_now, center_last, line_position, line_start, line_end):
    vehicle_is_online = False
    if center_now[1] >= line_start and \
        center_now[1] <= line_end and \
        center_last[1] >= line_start and \
        center_last[1] <= line_end:
        if (center_now[0] < line_position and \
            center_last[0] >= line_position) or \
            (center_now[0] > line_position and \
            center_last[0] <= line_position):
            vehicle_is_online = True

    return vehicle_is_online

def counting_vertical(center_now, center_last, line_position, line_start, line_end):
    vehicle_is_online = False
    if center_now[0] >= line_start and \
        center_now[0] <= line_end and \
        center_last[0] >= line_start and \
        center_last[0] <= line_end:
        if (center_now[1] < line_position and \
            center_last[1] >= line_position) or \
            (center_now[1] > line_position and \
            center_last[1] <= line_position):
            vehicle_is_online = True

    return vehicle_is_online
                
            