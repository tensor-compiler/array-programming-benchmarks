import numpy as np
import cv2
import os
import pytest
import matplotlib.pyplot as plt 

images_path = "./numpy/images"

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

def plot_image(img, img1, img2, xor_img, t1, t2):
    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img1, 'gray')
    ax[0, 0].title.set_text("Binned Image 1. t1 = " + str(t1))

    ax[0, 1].imshow(img2, 'gray')
    ax[0, 1].title.set_text("Binned Image 2. t2 = " + str(t2))

    ax[1, 0].imshow(img, 'gray')
    ax[1, 0].title.set_text("Saturdated Image")

    ax[1, 1].imshow(xor_img, 'gray')
    ax[1, 1].title.set_text("XOR Image")
    
    f.tight_layout()
    plt.show()

@pytest.mark.parametrize("t1", [100, 150, 200, 250])
def bench_edge_detection(tacoBench, t1, plot):
    images = load_dataset(images_path)
    
    sat_images = images[:,:,:,1]
    
    img = sat_images[0]
    
    t2 = t1 - 50
 
    bin_img1 = thresh(img, t1)
    bin_img2 = thresh(img, t2)
    num_elements = float(np.prod(bin_img1.shape))

    def bench():
        xor_img = np.logical_xor(bin_img1[0], bin_img2[0]).astype('int')
        return xor_img
    ret = tacoBench(bench)
    xor_img = bench()
    if plot:
        plot_image(img, bin_img1[0], bin_img2[0], xor_img, t1, t2)

    print("Sparsity img 1 ", np.sum(bin_img1 != 0) / num_elements)
    print("Sparsity img 2 ", np.sum(bin_img2 != 0) / num_elements)
    print("Sparsity xor ", np.sum(xor_img != 0) / num_elements)
    
if __name__=="__main__":
    main()

