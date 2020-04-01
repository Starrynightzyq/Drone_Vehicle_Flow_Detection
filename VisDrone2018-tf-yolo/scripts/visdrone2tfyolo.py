import os
import pandas as pd
from PIL import Image
import glob

import pdb

column_name = ['frame_index','target_id',
'bbox_left','bbox_top','bbox_width',
'bbox_height','score','object_category',
'truncation','occlusion']

def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        #此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        os.makedirs(path) 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

def visdrone2yolo(img_path, df):
	annotation = img_path
	for line in df.index:

		value_xmin = df['bbox_left'][line]
		value_ymin = df['bbox_top'][line]
		value_xmax = df['bbox_left'][line] + df['bbox_width'][line]
		value_ymax = df['bbox_top'][line] + df['bbox_height'][line]
		value_class_id = df['object_category'][line] # tensorflow-yolo 要求class编号从 0 开始

		# if not value_class_id in range(0,10):
		# 	print(img_path, value_class_id)

		annotation += ' ' + ','.join([str(value_xmin), str(value_ymin), str(value_xmax), str(value_ymax), str(value_class_id)])

	# print(annotation)
	# pdb.set_trace()

	return annotation
	

def main():

	# 用作 test 的图片数据的比例
	percentage_test = 10

	# 存放图片地址
	files_dir = os.path.abspath(os.path.join(os.getcwd(), '../', 'images'))

	#创建训练数据集和测试数据集：train.txt 和 test.txt
	file_train = open(os.path.abspath(os.path.join(os.getcwd(), '../', 'train.txt')), 'w')
	file_test = open(os.path.abspath(os.path.join(os.getcwd(), '../', 'test.txt')), 'w')

	image_dirs = []
	counter = 1
	counter_train = 0
	counter_test = 0
	index_test = round(100 / percentage_test)

	# 获取所有文件夹名称
	for root, dirs, files in os.walk(files_dir, topdown=False):
		for name in dirs:
			image_dirs.append(name)

	for dir in image_dirs:
		# mkdir(os.path.join('temp', dir))

		images = glob.glob(os.path.join(files_dir, dir, '*.jpg'))
		image_num = len(images)
		print('There are', image_num, 'images in', files_dir+'/'+dir)

		df = pd.read_csv(os.path.join(files_dir, dir+'.txt'))
		df.columns = column_name

		for image_ID in range(1, image_num):
			image_name = str(image_ID).zfill(7)+'.jpg'
			img_df = df[df['frame_index'] == image_ID]
			image_path = os.path.join(files_dir, dir, image_name)

			label_data = visdrone2yolo(image_path, img_df)

			if counter == index_test:
				counter = 1
				file_test.write(label_data + '\n')
				counter_test += 1
			else:
				file_train.write(label_data + '\n')
				counter = counter + 1
				counter_train += 1


	file_train.close()
	file_test.close()

	print('Processing done. There are', counter_train, 'images for training,', counter_test, 'images for testing.')


main()