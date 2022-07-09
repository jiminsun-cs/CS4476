from re import M
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
    
        self.indoor = None
        self.outdoor = None
        ###### START CODE HERE ######
        self.indoor = cv2.imread('indoor.png')
        self.outdoor = cv2.imread('outdoor.png')
        self.indoor = cv2.cvtColor(self.indoor, cv2.COLOR_BGR2RGB)
        self.outdoor = cv2.cvtColor(self.outdoor, cv2.COLOR_BGR2RGB) #to RGB
        ###### END CODE HERE ######

    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        # Load the images and plot their
        # R, G, B channels separately as grayscale images using matplotlib’s imshow() (use gray colormap). 
        # Then convert them into LAB color space using cv2.cvtColor() or skimage.color and plot the three channels again. 
        # Include the plots in your report. (Use function prob_4_1) [points - 7 Report]
        #        
        ###### START CODE HERE ######
        plt.subplot(131)
        plt.imshow(self.indoor[:,:,0], cmap='gray')
        plt.axis("off")
        plt.title("R_indoor")
        plt.subplot(132)
        plt.imshow(self.indoor[:,:,1], cmap='gray')
        plt.axis("off")
        plt.title("G_indoor")
        plt.subplot(133)
        plt.imshow(self.indoor[:,:,2], cmap='gray')
        plt.axis("off")
        plt.title("B_indoor")
        plt.show()

        plt.subplot(131)
        plt.imshow(self.outdoor[:,:,0], cmap='gray')
        plt.axis("off")
        plt.title("R_outdoor")
        plt.subplot(132)
        plt.imshow(self.outdoor[:,:,1], cmap='gray')
        plt.axis("off")
        plt.title("G_outdoor")
        plt.subplot(133)
        plt.imshow(self.outdoor[:,:,2], cmap='gray')
        plt.axis("off")
        plt.title("B_outdoor")
        plt.show()
        plt.axis("off")

        self.indoor = color.rgb2lab(self.indoor)
        self.outdoor = color.rgb2lab(self.outdoor)

        plt.subplot(131)
        plt.imshow(self.indoor[:,:,0], cmap='gray')
        plt.axis("off")
        plt.title("L_indoor")
        plt.subplot(132)
        plt.imshow(self.indoor[:,:,1], cmap='gray')
        plt.axis("off")
        plt.title("A_indoor")
        plt.subplot(133)
        plt.imshow(self.indoor[:,:,2], cmap='gray')
        plt.axis("off")
        plt.title("B_indoor")
        plt.show()

        plt.subplot(131)
        plt.imshow(self.outdoor[:,:,0], cmap='gray')
        plt.axis("off")
        plt.title("L_outdoor")
        plt.subplot(132)
        plt.imshow(self.outdoor[:,:,1], cmap='gray')
        plt.axis("off")
        plt.title("A_outdoor")
        plt.subplot(133)
        plt.imshow(self.outdoor[:,:,2], cmap='gray')
        plt.axis("off")
        plt.title("B_outdoor")
        plt.show()
        
        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        HSV = None
        ###### START CODE HERE ######
        input_img = cv2.imread('inputPS1Q4.jpg')
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # print(np.shape(input_img)) #(400, 600, 3)
        #do the necessary typecasting (double) and 
        # transform the image to values between [0,1] 
        # before performing the below operations.
        input_img = input_img / 255.0
        i_shape, j_shape, _= np.shape(input_img)
        HSV = np.zeros(np.shape(input_img))
        for i in range(i_shape):
            for j in range(j_shape):
                R = input_img[i,j,0]
                G = input_img[i,j,1]
                B = input_img[i,j,2]
                # print(np.shape(R))
                V = np.max([R, G, B])
                # print(V)
                # print(np.shape(V))
                min = np.min([R, G, B])
                C = V - min

                if V == 0: S = 0
                else: S = C / V
                
                if C == 0:
                    HSV[i,j,0] = 0
                elif V == R:
                    H_ = (G - B) / C
                elif V == G:
                    H_ = (B - R) / C + 2
                elif V == B:
                    H_ = (R - G) / C + 4

                if H_ < 0:
                    H = H_ / 6 + 1
                else: 
                    H = H_ / 6
                HSV[i,j,0] = H
                HSV[i,j,1] = S
                HSV[i,j,2] = V
        # plt.imshow(HSV)
        # plt.title("Final HSV Image")
        # plt.show()
        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()





