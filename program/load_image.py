#This is a program for loading data
#Author: Wenzhe Ouyang
#Start time: 2017/12/16

import cv2
import numpy as np
from matplotlib import pylab as plt

def loadimage(image_name):

	first_image = cv2.imread("./Dataset/" + image_name + "/" + image_name + "_01.png")
	number_images = 25
	height, width, channels = first_image.shape
	all_image = np.zeros([number_images,height,width,channels], dtype=np.float32)
	
	for image_index in range(number_images):
		present_image = cv2.imread("./Dataset/" + image_name + "/" + image_name \
			+ "_" + "%02d" %(image_index + 1)+ ".png")

		present_image = cv2.cvtColor(present_image,cv2.COLOR_BGR2RGB)
		all_image[image_index, :, :,: ] = np.array(present_image)
	
	all_image[:,:,:] = all_image[:,:,:] / 255
	data = np.r_[all_image[:,:,:,0],all_image[:,:,:,1],all_image[:,:,:,2]]
	
	all_same_position_pixel = np.zeros([height * width, number_images, 3])

	all_image_rec = np.zeros([number_images,height,width,channels], dtype=np.float32)

	for x in range(width):
		for y in range(height):
			all_same_position_pixel[y * width + x,:] = all_image[:,y,x,:]
	for x in range(width):
		for y in range(height):
			all_image_rec[:,y,x,:] = all_same_position_pixel[y * width + x,:] 

	#print(all_same_position_pixel.shape)
	#print(all_same_position_pixel.shape)
	#plt.imshow(all_image_rec[1])
	#plt.show()	
	return [all_same_position_pixel.astype(np.float32), width, height]
	
	#return all_image_roi, height, width
if __name__ == '__main__':
	loadimage("DragonsAndBunnies")