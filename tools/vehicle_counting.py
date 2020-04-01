import cv2
# import speed_prediction
import tools.speed_prediction as speed_prediction
# from core.config import cfg
# from core.utils import read_class_names

# is_vehicle_detected = [0]

left_vehicle_counter = 0
right_vehicle_counter = 0
bottom_vehicle_counter = 0

LEFT_INTERSECTION_ROI_POSITION = 400
LEFT_INTERSECTION_ROI_START = 300
LEFT_INTERSECTION_ROI_END = 550

RIGHT_INTERSECTION_ROI_POSITION = 1000
RIGHT_INTERSECTION_ROI_START = 0
RIGHT_INTERSECTION_ROI_END = 250

BOTTOM_INTERSECTION_ROI_POSITION = 600
BOTTOM_INTERSECTION_ROI_START = 600
BOTTOM_INTERSECTION_ROI_END = 1000

def vehicle_counting(
    box_top,
    box_bottom,
    box_right,
    box_left,
    current_frame_number,
    # crop_img,
    roi_position,
    roi_start,
    roi_end,
    roi_dirtction
    ):
    isInROI = False  # is the object that is inside Region Of Interest
    vehicle_front = 0
    is_vehicle_detected = []
    direction = ''
    speed = 0

    # direction change
    if roi_dirtction == 'top_to_bottom':
        vehicle_front = box_bottom
        vehicle_back = box_top
        vehicle_left = box_right
        vehicle_right = box_left
    elif roi_dirtction == 'bottom_to_top':
        vehicle_front = box_top
        vehicle_back = box_bottom
        vehicle_left = box_left
        vehicle_right = box_right
    elif roi_dirtction == 'left_to_right':
        vehicle_front = box_right
        vehicle_back = box_left
        vehicle_left = box_top
        vehicle_right = box_bottom
    elif roi_dirtction == 'right_to_left':
        vehicle_front = box_left
        vehicle_back = box_right
        vehicle_left = box_bottom
        vehicle_right = box_top
    else:
        vehicle_front = 0
        vehicle_back = 0
        vehicle_left = 0
        vehicle_right = 0
    
    # determine if the vehicle is on the line
    if max(vehicle_front,vehicle_back) > roi_position and \
        min(vehicle_front,vehicle_back) < roi_position and \
        vehicle_left > roi_start and \
        vehicle_left < roi_end and \
        vehicle_right > roi_start and \
        vehicle_right < roi_end:

        # # debug
        # print('vehicle on line')

        direction, speed, is_vehicle_detected, update_csv = speed_prediction.predict_speed(
            vehicle_front,
            vehicle_back,
            current_frame_number,
            # crop_img,
            roi_position)

    if(1 in is_vehicle_detected):
        counting = True
        del is_vehicle_detected[:]
        is_vehicle_detected = []
    else:
        counting = False

    return (direction, speed, counting)

def vehicle_counting_multi(
    image,
    box_top,
    box_bottom,
    box_right,
    box_left,
    current_frame_number):

    global left_vehicle_counter
    global right_vehicle_counter
    global bottom_vehicle_counter
    
    # left intersection
    left_direction, left_speed, left_counting = vehicle_counting(
        box_top,
        box_bottom,
        box_right,
        box_left,
        current_frame_number,
        LEFT_INTERSECTION_ROI_POSITION,
        LEFT_INTERSECTION_ROI_START,
        LEFT_INTERSECTION_ROI_END,
        'left_to_right')

    # when the vehicle passed over line and counted, make the color of ROI line green
    if left_counting:
        left_vehicle_counter += 1
        cv2.line(image, \
            (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_START), \
            (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_END), \
            (0, 0xFF, 0), 5)
    else:
        cv2.line(image, \
            (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_START), \
            (LEFT_INTERSECTION_ROI_POSITION, LEFT_INTERSECTION_ROI_END), \
            (0, 0, 0xFF), 5)
    
    # right intersection
    right_direction, right_speed, right_counting = vehicle_counting(
        box_top,
        box_bottom,
        box_right,
        box_left,
        current_frame_number,
        RIGHT_INTERSECTION_ROI_POSITION,
        RIGHT_INTERSECTION_ROI_START,
        RIGHT_INTERSECTION_ROI_END,
        'right_to_left')

    if right_counting:
        right_vehicle_counter += 1
        cv2.line(image, \
            (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_START), \
            (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_END), \
            (0, 0xFF, 0), 5)
    else:
        cv2.line(image, \
            (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_START), \
            (RIGHT_INTERSECTION_ROI_POSITION, RIGHT_INTERSECTION_ROI_END), \
            (0, 0, 0xFF), 5)
    
    # bottom intersection
    bottom_direction, bottom_speed, bottom_counting = vehicle_counting(
        box_top,
        box_bottom,
        box_right,
        box_left,
        current_frame_number,
        BOTTOM_INTERSECTION_ROI_POSITION,
        BOTTOM_INTERSECTION_ROI_START,
        BOTTOM_INTERSECTION_ROI_END,
        'bottom_to_top')

    if bottom_counting:
        bottom_vehicle_counter += 1
        cv2.line(image, \
            (BOTTOM_INTERSECTION_ROI_START, BOTTOM_INTERSECTION_ROI_POSITION), \
            (BOTTOM_INTERSECTION_ROI_END, BOTTOM_INTERSECTION_ROI_POSITION), \
            (0, 0xFF, 0), 5)
    else:
        cv2.line(image, \
            (BOTTOM_INTERSECTION_ROI_START, BOTTOM_INTERSECTION_ROI_POSITION), \
            (BOTTOM_INTERSECTION_ROI_END, BOTTOM_INTERSECTION_ROI_POSITION), \
            (0, 0, 0xFF), 5)

    return (image, left_vehicle_counter, right_vehicle_counter, bottom_vehicle_counter)




# def draw_bbox(image, bboxes, current_frame_number, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
#     """
#     bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
#     """

#     num_classes = len(classes)
#     image_h, image_w, _ = image.shape
#     hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
#     colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#     colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

#     random.seed(0)
#     random.shuffle(colors)
#     random.seed(None)

#     for i, bbox in enumerate(bboxes):
#         coor = np.array(bbox[:4], dtype=np.int32)
#         fontScale = 0.5
#         score = bbox[4]
#         class_ind = int(bbox[5])
#         bbox_color = colors[class_ind]
#         bbox_thick = int(0.6 * (image_h + image_w) / 600)
#         c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
#         cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

#         image, left_vehicle_counter, right_vehicle_counter, bottom_vehicle_counter = vehicle_counting_multi(image, \
#             coor[1], \
#             coor[3], \
#             coor[2], \
#             coor[0], \
#             current_frame_number)

#         if show_label:
#             bbox_mess = '%s: %.2f' % (classes[class_ind], score)
#             t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
#             cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

#             cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

#     left_info = 'Left Vehicle Number: ' + left_vehicle_counter
#     right_info = 'Right Vehicle Number: ' + right_vehicle_counter
#     bottom_info = 'Bottom Vehicle Number: ' + bottom_vehicle_counter

#     cv2.putText(image, text=left_info, org=(50, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 0, 0), thickness=2)
#     cv2.putText(image, text=right_info, org=(50, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 0, 0), thickness=2)
#     cv2.putText(image, text=bottom_info, org=(50, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=1, color=(255, 0, 0), thickness=2)

#     return image


