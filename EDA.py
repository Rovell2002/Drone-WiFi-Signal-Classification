import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load transformed data
filename1 = r'C:\Users\lolaf\Desktop\Thesis_code\Resized_128.h5'
h5f = h5py.File(filename1, 'r')
X_train = np.array(h5f['X_train'])
Y_train = np.array(h5f['y_train'])
X_test = np.array(h5f['X_test'])
Y_test = np.array(h5f['y_test'])
h5f.close()

print("--" * 10)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
print("Training labels size:", Y_train.shape)
print("Testing labels size:", Y_test.shape)
print("--" * 10)

def plot_sample_spectrograms(X, num_samples=20):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
    
    for i in range(num_samples):
        axes[i].imshow(X[i], aspect='auto', origin='lower')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def compute_statistics(X):
    pixel_values = X.flatten()
    mean_intensity = np.mean(pixel_values)
    std_intensity = np.std(pixel_values)
    return mean_intensity, std_intensity

def plot_intensity_distribution(X):
    pixel_values = X.flatten()
    plt.hist(pixel_values, bins=50, color='blue', edgecolor='black')
    plt.title('Distribution of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def plot_average_spectrogram(X):
    avg_spectrogram = np.mean(X, axis=0)
    plt.imshow(avg_spectrogram, aspect='auto', origin='lower')
    plt.title('Average Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity')
    plt.show()

plot_sample_spectrograms(X_train)

mean_intensity_train, std_intensity_train = compute_statistics(X_train)
mean_intensity_test, std_intensity_test = compute_statistics(X_test)

print(f'Mean Intensity (Train): {mean_intensity_train}')
print(f'Standard Deviation of Intensity (Train): {std_intensity_train}')
print(f'Mean Intensity (Test): {mean_intensity_test}')
print(f'Standard Deviation of Intensity (Test): {std_intensity_test}')

plot_intensity_distribution(X_train)

plot_average_spectrogram(X_train)

def plot_label_distribution(y_train, y_test):
    plt.figure(figsize=(12, 6))
    sns.countplot(y_train)
    plt.title('Training Set Label Distribution')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(y_test)
    plt.title('Testing Set Label Distribution')
    plt.show()

plot_label_distribution(Y_train, Y_test)

def plot_sample_spectrograms_with_labels(X, y, num_samples=3):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    
    for i in range(num_samples):
        axes[i].imshow(X[i], aspect='auto', origin='lower')
        axes[i].set_title(f'Sample {i+1}\nLabel: {y[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_sample_spectrograms_with_labels(X_train, Y_train)