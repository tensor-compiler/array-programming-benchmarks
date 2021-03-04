import numpy as np
import cv2
import os
import pytest 

images_path = "./images"

def load_dataset(image_folder):
    files = sorted(os.listdir(image_folder))
    images = []
    for f in files:
        path = os.path.join(image_folder, f)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        images.append(img)
    
    images =  np.stack(images, axis=0)
    return images

def thresh(images, t=85):
    if len(images.shape) < 3:
        images = np.expand_dims(images, axis=0)
    thresh_imgs = []
    for i in range(images.shape[0]):
        img = images[i]
        ret, thresh_img = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        thresh_imgs.append(thresh_img)

    thresh_imgs =  np.stack(thresh_imgs, axis=0)        
    return thresh_imgs 

def plot_image(image, title=""):
    plt.imshow(image, 'gray')
    plt.title(title)
    plt.show()

def main():
    images = load_dataset(images_path)
    print(images.shape)
    sat_images = images[:,:,:,1]
    print(sat_images.shape)     
    img = sat_images[0]
    print(img.shape)
    plot_image(img)
    t1 = 150
    t2 = 100
    bin_img1 = thresh(img, t1)
    bin_img2 = thresh(img, t2)
    print(bin_img1.shape)

    print("nnz img 1 ", np.sum(bin_img1 != 0))
    print("nnz img 1 ", np.sum(bin_img2 != 0))
    print("total arr size ", np.prod(bin_img1.shape))

    plot_image(bin_img1[0], "thresh = " + str(t1))
    plot_image(bin_img2[0], "thresh = " + str(t2))

    xor_img = np.logical_xor(bin_img1[0], bin_img2[0]).astype('int')
    plot_image(xor_img)

    print("nnz xor ", np.sum(xor_img != 0))
    
if __name__=="__main__":
    main()

