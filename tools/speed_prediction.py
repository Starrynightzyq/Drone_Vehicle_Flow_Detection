#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
# from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
front_position_of_detected_vehicle = [0]

def predict_speed(
    vehicle_front,
    vehicle_back,
    current_frame_number,
    # crop_img,
    roi_position
    ):
    speed = 'n.a.'  # means not available, it is just initialization
    direction = 'n.a.'  # means not available, it is just initialization
    scale_constant = 1  # manual scaling because we did not performed camera calibration
    isInROI = True  # is the object that is inside Region Of Interest
    update_csv = False

    # speed scale
    # if vehicle_front < 250:
    #     scale_constant = 1  # scale_constant is used for manual scaling because we did not performed camera calibration
    # elif vehicle_front > 250 and vehicle_front < 320:
    #     scale_constant = 2  # scale_constant is used for manual scaling because we did not performed camera calibration
    # else:
    #     isInROI = False
    #     print('isInROI', isInROI)
    
    # if len(front_position_of_detected_vehicle) != 0 \
    #     and vehicle_front - front_position_of_detected_vehicle[0] > 0 \
    #     and 205 < front_position_of_detected_vehicle[0] \
    #     and front_position_of_detected_vehicle[0] < 210 \
    #     and roi_position < vehicle_front:
    #     is_vehicle_detected.insert(0, 1)
    #     update_csv = True

    if len(front_position_of_detected_vehicle) != 0 \
        and vehicle_front - front_position_of_detected_vehicle[0] > 0 \
        and abs((vehicle_front + vehicle_back)/2 - roi_position) < 2:
        is_vehicle_detected.insert(0, 1)
        update_csv = True
        # image_saver.save_image(crop_img)  # save detected vehicle image

    # debug
    print('vehicle_front', vehicle_front)

    # for debugging
    # print("front_position_of_detected_vehicle[0]: " + str(front_position_of_detected_vehicle[0]))
    # print("vehicle_front: " + str(vehicle_front))
    if vehicle_front > front_position_of_detected_vehicle[0]:
        direction = 'forward'
    else:
        direction = 'backward'

    if isInROI:
        pixel_length = vehicle_front - front_position_of_detected_vehicle[0]
        scale_real_length = pixel_length * 44  # multiplied by 44 to convert pixel length to real length in meters (chenge 44 to get length in meters for your case)
        total_time_passed = current_frame_number - current_frame_number_list[0]
        scale_real_time_passed = total_time_passed * 24  # get the elapsed total time for a vehicle to pass through ROI area (24 = fps)
        if scale_real_time_passed != 0:
            speed = scale_real_length / scale_real_time_passed / scale_constant  # performing manual scaling because we have not performed camera calibration
            speed = speed / 6 * 40  # use reference constant to get vehicle speed prediction in kilometer unit
            current_frame_number_list.insert(0, current_frame_number)
            front_position_of_detected_vehicle.insert(0, vehicle_front)

    return (direction, speed, is_vehicle_detected, update_csv)




