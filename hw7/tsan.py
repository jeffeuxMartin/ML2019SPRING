import os
import sys
from time import ctime
import numpy as np 
from skimage.io import imread, imsave

"""
bash pca.sh <images path> <input image> <reconstruct image>
e.g. bash  pca.sh  Aberdeen/   87.jpg   87_reconstruct.jpg

python3 $1 $2 $3
"""

images_path = sys.argv[1]
input_images = sys.argv[2]
reconstruct_image = sys.argv[3]

IMAGE_PATH = 'Aberdeen'

# Images for compression & reconstruction
test_image = ['12.jpg','18.jpg','42.jpg','57.jpg','73.jpg'] 

# Number of principal components used
k = 5




def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M




IMAGE_PATH = images_path





filelist = os.listdir(IMAGE_PATH) 

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())




training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 





print(ctime())
# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data, full_matrices = False)  
print(ctime())






for x in test_image: 
    # Load image & Normalize
    picked_img = imread(os.path.join(IMAGE_PATH, x))  
    X = picked_img.flatten().astype('float32') 
    X -= mean
    
    # Compression
    weight = np.array([v[i].dot(X) for i in range(k)])  
    
    # Reconstruction
    reconstruct = process(weight.dot(v[:k]) + mean)
    imsave(x[:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape)) 

picked_img = imread(input_images)  
X = picked_img.flatten().astype('float32') 
X -= mean

# Compression
weight = np.array([v[i].dot(X) for i in range(k)])  

# Reconstruction
reconstruct = process(weight.dot(v[:k]) + mean)
imsave(reconstruct_image, reconstruct.reshape(img_shape)) 


average = process(mean)
imsave('average.jpg', average.reshape(img_shape))  




for x in range(10):
    eigenface = process(1-v[x]).reshape(img_shape)
    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  






for i in range(5):
    number = s[i] * 100 / sum(s)
    print(round(number, 1), end='%, ')