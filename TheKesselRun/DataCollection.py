import numpy as np
import glob
import scipy.fftpack as fft
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.feature as skiff
import skimage.color as skic

# Do shit

# fft.fft2()

# 1610.31 Hz framerate
# Scan velocity: 2850 Hz

class HTRx():
    def __init__(self, debug=False):
        # Todo Image directory or single image?
        self.img_pth = None
        self.cali_pth = None
        self.img = None
        self.rawdata = None

        if debug is True:
            self.img_pth = r"C:\Users\quick\Desktop\DataCleanup\NH3calib2-validation-3.5-1.img"
            self.cali_pth = None

    def load_image(self):
        fid = open(self.img_pth, 'rb')
        self.rawdata = np.fromfile(fid, np.dtype('>u2')) # big-endian unsigned integer (16bit)

    def process_image(self):
        if self.rawdata is None:
            print('No raw data loaded.')
            exit()

    def check_img_exists(self):
        if self.img is None:
            print('Image does not exist.')
            exit()

    def clip_threshold(self, ax=None, lowth=0.0, highth=1.0):
        if ax is None:
            fig, ax = plt.subplots(nrows=2)
            ax.imshow(self.img[:,:,0], clim=(lowth, highth))
            plt.show()
        else:
            ax.imshow(self.img[:,:,0], clim=(lowth, highth))

    def plot_histogram(self, ax=None):
        self.check_img_exists()

        if ax is None:
            plt.hist(self.img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
            plt.show()
        else:
            ax.hist(self.img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

    def examine_data(self):
        self.load_image()

        fig, ax = plt.subplots(nrows=3, ncols=2)
        # 1. Plot the original image
        ax[0,0].imshow(self.img)

        # 2. Plot a histogram of the image
        self.plot_histogram(ax=ax[0,1])

        # 3. Plot a colored image and thresholded image
        self.clip_threshold(ax=ax[1,0])
        self.clip_threshold(ax=ax[1,1], lowth=0.2, highth=1.0)

        # print()
        # skiff.blob_dog(self.img[:, :, 0])
        ax[2,0].imshow(skiff.blob_dog(skic.rgb2gray(self.img)))

        plt.show()

    def process_data(self):
        pass

    def write_data(self):
        pass

    def load_calibration_file(self):
        pass

if __name__ == '__main__':
    llama = HTRx(debug=True)
    llama.load_image()
    print(llama.rawdata)
    print(len(llama.rawdata))
    print(llama.rawdata[:-82176])
    image = llama.rawdata[82176:].reshape((128, 128, 1920, 32))
    # image = llama.rawdata[82176:].reshape((1920, 128, 128, 32))
    # print(image.shape)
    plt.imshow(image[:,:,7,0])
    plt.colorbar()
    plt.show()

    # plt.plot(image[0,0,:,0])
    # plt.show()
    # df = pd.DataFrame(image)
    #
    # ave_array = np.array([0])
    # for i in range(1920):
    #     df_cut = df[i*128:(i+1)*128].copy()
    #
    #     if i == 0:
    #         ave_array = df_cut.values
    #     else:
    #         ave_array = np.mean(np.array([ave_array, df_cut.values]), axis=0)
    #
    # print(ave_array)

    # plt.imshow(ave_array)
    # plt.show()

