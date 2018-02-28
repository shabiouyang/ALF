#Author: Wenzhe Ouyang
#Strating time: 2017/12/25

import numpy as np
import load_image
from sklearn.decomposition import PCA
import cv2

def get_train_data():
	data1, width1, height1 = load_image.loadimage("Knights")
	data2, width2, height2 = load_image.loadimage("Bunny")
	data3, width3, height3 = load_image.loadimage("Amethyst")
	data4, width4, height4 = load_image.loadimage("Jelly Beans")
	data5, width5, height5 = load_image.loadimage("dice")

	data_1 = np.r_[data1[:,:,0],data1[:,:,1],data1[:,:,2]]
	data_2 = np.r_[data2[:,:,0],data2[:,:,1],data2[:,:,2]]
	data_3 = np.r_[data3[:,:,0],data3[:,:,1],data3[:,:,2]]
	data_4 = np.r_[data4[:,:,0],data4[:,:,1],data4[:,:,2]]
	data_5 = np.r_[data4[:,:,0],data4[:,:,1],data4[:,:,2]]


	train_data = np.r_[data_1, data_2, data_3, data_4, data_5]
	train_length = len(train_data)
	return train_data, train_length

def get_test_data():
	test, width, height = load_image.loadimage("DragonsAndBunnies")
	test_data = np.r_[test[:,:,0], test[:,:,1], test[:,:,2]]
	test_length = len(test_data)
	return test_data, test_length, width, height


def main():
	train_data, train_length = get_train_data()
	test_data, test_length, width, height = get_test_data()

	model = PCA(n_components=5)
	W = model.fit_transform(train_data)
	H = model.components_
	compressed_images = model.transform(test_data)
	output_images = model.inverse_transform(compressed_images)

	output_length= len(output_images)
	rgb_length = int(output_length/3)
	reconstruct_subimages = np.zeros([height*width, 25, 3], dtype=np.float32)
	for channels in range(3):
		reconstruct_subimages[:, :, int(channels)] = output_images[(rgb_length*channels):(rgb_length*(channels+1)),:]
	all_image_rec = np.zeros([25,height,width,3], dtype=np.float32)
	for x in range(width):
		for y in range(height):
			all_image_rec[:,y,x,:] = reconstruct_subimages[y * width + x,:]*255
	for numbers in range(25):
		all_image_rec[numbers, :, :, :] = cv2.cvtColor(all_image_rec[numbers, :, :, :], cv2.COLOR_BGR2RGB)
		cv2.imwrite("./result/pca_5shots/" + str(numbers+1) + "_" + "PCA" + ".png", all_image_rec[numbers, :, :, :])
if __name__ == '__main__':
	main()