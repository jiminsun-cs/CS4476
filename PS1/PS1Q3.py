from pickletools import float8
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = None
        ###### START CODE HERE ######
        self.img = cv2.imread('inputPS1Q3.jpg')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        ###### END CODE HERE ######
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image
        """
        gray = None
        ###### START CODE HERE ######
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ###### END CODE HERE ######
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        swapImg = None
        ###### START CODE HERE ######
        # print(np.shape(self.img)) #900 600 3
        swapImg = np.ones(np.shape(self.img))

        imgHolder = np.copy(self.img)
        red = np.copy(self.img[:,:,0])
        green = np.copy(self.img[:,:,1])
        imgHolder[:,:,0] = green
        imgHolder[:,:,1] = red
        swapImg = imgHolder
        plt.imshow(swapImg)
        plt.title("Q3.1")
        plt.show()
        ###### END CODE HERE ######
        return swapImg

    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        grayImg = None
        ###### START CODE HERE ######  
        grayImg = self.rgb2gray(self.img)
        plt.imshow(grayImg, cmap='gray')
        plt.title("Q3.2")
        plt.show()

        ###### END CODE HERE ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        negativeImg = None
        ###### START CODE HERE ######
        x, y = np.shape(self.prob_3_2())
        template = np.ones((x,y)) * 255
        negativeImg = template - self.prob_3_2()
        plt.imshow(negativeImg, cmap='gray')
        plt.title("Q3.3")
        plt.show()
        ###### END CODE HERE ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        mirrorImg = None
        ###### START CODE HERE ######
        # flip it left to right
        mirrorImg = self.prob_3_2()
        mirrorImg = mirrorImg[:, ::-1]
        plt.imshow(mirrorImg, cmap='gray')
        plt.title("Q3.4")
        plt.show()
        ###### END CODE HERE ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        avgImg = None
        ###### START CODE HERE ######
        grayscale = np.array(self.prob_3_2(), dtype = float)
        x,y = np.shape(grayscale)
        mirror = np.array(self.prob_3_4(), dtype = float)
        avgImg = (grayscale + mirror)/ 2
        avgImg = avgImg.astype(np.int)
        plt.imshow(avgImg, cmap='gray')
        plt.title("Q3.5")
        plt.show()
        ###### END CODE HERE ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            noisyImg, noise: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
            and the noise
        """
        noisyImg, noise = [None]*2
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        grayImg = np.array(grayImg, dtype = np.float)
        
        np.save('./noise.npy', noise)
        i, j = np.shape(grayImg)
        noise = np.random.rand(i,j)
        noise = np.ceil(noise * 256) - 1
        grayImg = grayImg + noise
        grayImg[grayImg > 255] = 255
        noisyImg = grayImg
        plt.imshow(noisyImg, cmap='gray')
        plt.title("Q3.6")
        plt.show()

        ###### END CODE HERE ######
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()

    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()

    




