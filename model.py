import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

"""
Read images and steering measurements.
"""
lines = []
with open('.\\data\\driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Split validation data.
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

# Define generator function.
def generator(lines, batch_size=64, correction_factor=0.3):
	num_lines = len(lines)
	while 1:
		shuffle(lines)
		for offset in range(0, num_lines, batch_size):
			batch_lines = lines[offset:offset+batch_size]
		
			images = []
			angles = []	
			for batch_line in batch_lines:
				# Read center image data.
				center_im_path = '.\\data\\IMG\\' + batch_line[0].split('\\')[-1]
				center_image = cv2.cvtColor(cv2.imread(center_im_path), cv2.COLOR_BGR2GRAY) # Read and convert to grayscale.
				center_angle = float(batch_line[3])
				images.append(center_image)
				angles.append(center_angle)
				# Read left image data.
				left_im_path = '.\\data\\IMG\\' + batch_line[1].split('\\')[-1]
				left_image = cv2.cvtColor(cv2.imread(left_im_path), cv2.COLOR_BGR2GRAY) # Read and convert to grayscale.
				left_angle = float(batch_line[3]) + correction_factor # Correct angle for left camera image.
				images.append(left_image)
				angles.append(left_angle)
				# Read right image data.
				right_im_path = '.\\data\\IMG\\' + batch_line[2].split('\\')[-1]
				right_image = cv2.cvtColor(cv2.imread(right_im_path), cv2.COLOR_BGR2GRAY) # Read and convert to grayscale.
				right_angle = float(batch_line[3]) - correction_factor # Correct angle for right camera image.
				images.append(right_image)
				angles.append(right_angle)
				
				# Augment Data.
				augmented_images, augmented_angles = [], []
				for image, angle in zip(images, angles):
					# Add original image and its angle.
					augmented_images.append(image)
					augmented_angles.append(angle)
					# Add darkenened (low brightness) image and its angle.
					augmented_images.append(image*0.1)
					augmented_angles.append(angle)
					# Add flipped image and its flipped angle.
					augmented_images.append(cv2.flip(image, 1))
					augmented_angles.append(angle*-1)
		
			x_train = np.expand_dims(np.array(augmented_images), axis=3) # Add 4th dimension since these are grayscale images.
			y_train = np.array(augmented_angles)
			yield shuffle(x_train, y_train)

# Define train and validation generators.
train_generator = generator(train_lines)		
validation_generator = generator(validation_lines)	
	
"""
Build and train model using Keras.
"""
# Define model.
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 1))) # Crop unnecessary parts of the images.
model.add(Lambda(lambda x: x / 255.0 - 0.5)) # Normalize images.
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train model.
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_lines), \
validation_data=validation_generator, nb_val_samples=len(validation_lines), nb_epoch=30)

# Save model.
model.save('model.h5')

# Plot training and validation loss.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()