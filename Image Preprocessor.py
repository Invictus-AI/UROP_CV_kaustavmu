import cv2
import numpy as np
import glob

#Function names should be the names of the folders. Dataset folders should follow the template, being dataset(template), template replaced with the function name.

def canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def gs(image):
    # Already gray!
    return image

def gshe(image):
    # Histogram Equalization on grayscale image.
    return cv2.equalizeHist(image)

def he(image):
    # Isolates luminosity layer and applies histogram equalization on it.
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

def agt(image):
    # Gaussian adaptive thresholding.
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)

functions = [canny, gs, gshe, he, agt]
grayscale = [True, True, True, False, True]

#Don't touch anything below this!

folders = [str(i).split(' ')[1] for i in functions]

for folder in ['train', 'val']:
    for file in glob.glob("./dataset/images/" + folder + "/*.jpg"):
        filename = file[(18 + len(folder)):-4]
        colorimage = cv2.imread(file)
        grayimage = cv2.imread(file, 0)
        for i in range(len(functions)):
            function = functions[i]
            foldername = folders[i]
            grays = grayscale[i]
            cv2.imwrite('./dataset(' + foldername + ')/images/' + folder + '/' + filename + '.png', function(grayimage if grays else colorimage))
        
