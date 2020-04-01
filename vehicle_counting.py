import pickle
import cv2
# 进度条
from tqdm import tqdm

from tools.trackers_to_perframe import draw_bbox_with_counting

pickle_file_path = 'output/test_output/tmp.pk'
video_path = "./docs/images/1.mp4"
video_output_path = './output/counting.mp4'

showVideo_flag = False
writeVideo_flag = True

with open(pickle_file_path, 'rb') as pk_f:
    trackers = pickle.load(pk_f)
    print('=> load trackers from pk file .')

vid = cv2.VideoCapture(video_path)
if vid.isOpened():
    # 总帧数
    total_frame_counter = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
else:
    raise IOError("Couldn't open webcam or video")

if writeVideo_flag:
    video_FourCC = cv2.VideoWriter_fourcc(*'MP4V')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if video_output_path != "":
        video_out = cv2.VideoWriter(video_output_path, video_FourCC, video_fps, video_size)

# 创建进度条
if not showVideo_flag:
    pbar = tqdm(total=total_frame_counter)

while True: 
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

    result = draw_bbox_with_counting(frame, vid.get(1), trackers)

    if showVideo_flag:
        cv2.imshow("result", result)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
    else:
        pbar.update(1)

    if writeVideo_flag:
        # save a frame
        video_out.write(result)

# Release everything if job is finished
video_out.release()
cv2.destroyAllWindows()   
pbar.close() 




