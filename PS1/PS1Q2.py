from re import S
import numpy as np
import matplotlib.pyplot as plt

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        self.A = np.load('inputAPS1Q2.npy')
        ###### END CODE HERE ######
        pass
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######
        A = self.A
        A = np.array(A)
        A = A.reshape((1,-1))
        A = np.sort(A)
        A = A[:, ::-1]
        plt.imshow(A, cmap = 'gray', aspect='auto')
        plt.title("Q2.1")
        plt.show()
        ###### END CODE HERE ######
        return
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        A = self.A
        plt.hist(A.flatten(), bins = 20)
        plt.title("Q2.2")
        plt.show()
        ###### END CODE HERE ######
        return
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        X = None
        ###### START CODE HERE ######
        length = np.shape(self.A)[0]
        X = self.A[length//2:, :length//2]
        ###### END CODE HERE ######
        return X
    
    def prob_2_4(self):

        Y = None
        ###### START CODE HERE ######
        mean_intensity = np.mean(self.A)
        newA = self.A - mean_intensity
        Y = newA
        ###### END CODE HERE ######
        return Y
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        Z = None
        ###### START CODE HERE ######
        i, j= np.shape(self.A)
        t = np.mean(self.A)
        new_A = np.zeros((i,j,3)) #r,g,b
        new_A[self.A > t, 0] = 1 # in the R channel, where above t -> set to red
        new_A[self.A > t, 1] = 0 # in the R channel, where above t -> set to red
        new_A[self.A > t, 2] = 0 # in the R channel, where above t -> set to red 
        Z = new_A
        # plt.imshow(new_A)
        # plt.show()
        ###### END CODE HERE ######
        return Z


if __name__ == '__main__':
    
    p2 = Prob2()
    
    p2.prob_2_1()
    p2.prob_2_2()
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()