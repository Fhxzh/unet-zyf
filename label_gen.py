from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import numpy as np
import os
import glob
from PIL import Image

class RGB_2_Gray(object):
	def __init__(self,data_path="test_demo/data/train/image"):
		self.data_path=data_path
	def create_gray(self):
		print('-----------------------------')
		print('create gray images...')
		print('-----------------------------')

		imgs=glob.glob(self.data_path+'/*.'+'jpg')
		print(len(imgs))

		for i in range(len(imgs)):
			imgname=imgs[i]
			img=Image.open(imgname)
			Lim=img.convert("L")

			Lim.save(self.data_path+"/"+"%d.tif"%(i))
			os.remove(imgname)

class label_generator(object):
	def __init__(self,rows,cols,data_path="test_demo/data/train/image",out_path="test_demo/data/train/label"):
		self.rows=rows
		self.cols=cols
		self.data_path=data_path
		self.out_path=out_path

	def createLabel(self):
		
		print('-----------------------------')
		print('create labels...')
		print('-----------------------------')

		imgs=glob.glob(self.data_path+'/*.'+'jpg')
		print(len(imgs))

		for i in range(len(imgs)):
			imgname=imgs[i]
			bim=np.zeros((self.rows,self.cols), dtype='uint8')

			img=Image.open(imgname)
			r,g,b=img.split()
			r_2_arr=img_to_array(r)
			g_2_arr=img_to_array(g)
			b_2_arr=img_to_array(b)

			for rowIndex in range(self.rows):
				for colIndex in range(self.cols):
					if((r_2_arr[rowIndex][colIndex]>=234 and r_2_arr[rowIndex][colIndex]<=250) and r_2_arr[rowIndex][colIndex]!=g_2_arr[rowIndex][colIndex]):
						bim[rowIndex][colIndex]=255
					else:
						bim[rowIndex][colIndex]=0

			bim=Image.fromarray(bim)
			bim.save(self.out_path+"/"+"%d.tif"%(i))


if __name__== "__main__":
	mylabel=label_generator(512, 512)
	mylabel.createLabel()

	# myRGB2Gray=RGB_2_Gray()
	# myRGB2Gray.create_gray()
