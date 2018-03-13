# Read IMG File

import matplotlib.pyplot as plt
import numpy as np

# Parameters.
input_filename = r"C:\Users\quick\Desktop\NH3calib2-validation-3.5-1.img"
shape = (31459848, 32) # matrix size
dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
output_filename = "JPCLN001.PNG"

# Reading.
fid = open(input_filename, 'rb')
data = np.fromfile(fid, dtype)
fid.close()
image = data.reshape(shape)

# Display.
plt.imshow(image, cmap = "gray")
plt.savefig(output_filename)
plt.show()