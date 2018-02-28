# Autoencoder-for-Light-Field
Python implementation for Light field Autoencoder, it can be used to light field compression or coded aperture camera.

# Prerequisites
* Tensorflow 1.4+
* OpenCV 3.1+
* sklearn 

# Dataset
I'm using Stanford Light Field Dataset. I have uploaded part of this dataset(5*5) in this reposity, you can use it directly. 
If you want try to train more dataset, you can download on http//lightfield.stanford.edu/lfs.html.

# How to run the code
* Run the main.py to get reconstrution images.
* Run the evaluation.py to calculate PSNR.
* Run the NMF.py or PCA.py to get reconstruction images using NMF method or PCA method, and you can campared it with Autoencoder method.
