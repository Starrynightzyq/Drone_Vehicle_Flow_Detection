import cv2
import numpy as np

# 直方图均衡
def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    # print len(channels)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# 图像预处理的函数
# 调整亮度和对比度
# c:对比度, b:亮度
def contrast_brightness_image(img, c, b):
    h, w, ch = img.shape  # 获取shape的数值，height/width/channel
    # 新建全零图片数组blank,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    blank = np.zeros([h, w, ch], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b) # 计算两个图像阵列的加权和 dst=src1*alpha+src2*beta+gamma
    return dst