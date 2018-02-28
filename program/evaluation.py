import math
import cv2
import numpy as np
def psnr_gray(img1, img2, width, height):   
    sum = 0;
    for i in range (height):
        for j in range(width):
            sum += (img1[i,j] - img2[i,j]) * (img1[i,j] - img2[i,j])
    mse = sum / (width * height)
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr

def psnr_color(img1, img2, width, height):    
    sum = 0;
    for i in range (height):
        for j in range(width):
            for k in range(3):
                sum += (img1[i,j,k] - img2[i,j,k]) * (img1[i,j,k] - img2[i,j,k])
    mse = sum / (3*width*height)
    psnr = 20 * math.log10(255/math.sqrt(mse))
    return psnr

first_image = cv2.imread("./result/pca_5shots/1_autoencoder.png")
height, width, channels = first_image.shape
all_image = np.zeros([5*height, 5*width, channels], dtype=np.float32)
original = cv2.imread("./Dataset/DragonsAndBunnies/All_DragonsAndBunnies.png")
src_height, src_width, src_channels = original.shape
all_original = np.zeros([src_height, src_width, src_channels], dtype=np.float32)
for x in range(src_height):
    for y in range(src_width):
        for chan in range(src_channels):
            all_original[x, y, chan] = original[x, y, chan]
for i in range(25):
    present_image = cv2.imread("./result/pca_5shots/" + "%d" %(i+1) + "_" + "autoencoder" + ".png")
    rows = int(i/5)
    cols = i%5
    all_image[(rows*height):((rows+1)*height), (cols*width):((cols+1)*width),: ] = np.array(present_image)

print(all_original.dtype, all_image.dtype)
psnr = psnr_color(all_image, all_original, 5*width, 5*height)
print(psnr)