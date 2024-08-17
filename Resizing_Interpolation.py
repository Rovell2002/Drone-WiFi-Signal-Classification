import h5py
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import pdb

# Replace 'path_to_your_file.h5' with the actual path to your HDF5 file
h5_file_path = r'C:\Users\lolaf\Desktop\Thesis_code\Data\GenDataRSNT_10_snrss2_NW.h5'
new_h5_file_path = r'C:\Users\lolaf\Desktop\Thesis_code\Resized_128.h5'

# Open the HDF5 file
with h5py.File(h5_file_path, 'r') as h5_file:
    # Assuming 'X_train', 'X_test', 'y_train', 'y_test' are the dataset names in your HDF5 file
    X_train = h5_file['X_train'][:]
    X_test = h5_file['X_test'][:]
    y_train = h5_file['labels_RSNT_F_train'][:]
    y_test = h5_file['labels_RSNT_F_test'][:]

    # Convert the pixel values from the range [0, 1] to [0, 255]
    X_train_255 = (X_train * 255).astype(np.float32)
    X_test_255 = (X_test * 255).astype(np.float64)

    # Resize the images to 128x128 using different interpolation methods
    X_train_128 = np.array([resize(image, (128, 128), preserve_range=True, order=5).astype(np.float32) for image in X_train_255])  
    X_test_128 = np.array([resize(image, (128, 128), preserve_range=True, order=5).astype(np.float64) for image in X_test_255])  

    # Print the min and max pixel values of the 128x128 image dataset
    print(f"Minimum pixel value in X_train_128: {np.min(X_train_128)}")
    print(f"Maximum pixel value in X_train_128: {np.max(X_train_128)}")
    print(f"Minimum pixel value in X_test_128: {np.min(X_test_128)}")
    print(f"Maximum pixel value in X_test_128: {np.max(X_test_128)}")

    # Visual inspection of one example
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(X_train_255[0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Resized Image")
    plt.imshow(X_train_128[0], cmap='gray')
    plt.show()

    # Normalize the 128x128 images to [0, 1] range
    X_train_128_normalized = X_train_128 / 255.
    X_test_128_normalized = X_test_128 / 255.

    # Save the normalized dataset along with labels to a new HDF5 file
    with h5py.File(new_h5_file_path, 'w') as new_h5_file:
        new_h5_file.create_dataset('X_train', data=X_train_128_normalized)
        new_h5_file.create_dataset('X_test', data=X_test_128_normalized)
        new_h5_file.create_dataset('y_train', data=y_train)
        new_h5_file.create_dataset('y_test', data=y_test)

    print(f"New dataset saved to {new_h5_file_path}")