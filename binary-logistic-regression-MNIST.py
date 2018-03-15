from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils  

import matplotlib.pyplot as plt
import numpy as np

## Function definitions
def plot_metrics(history):
	plt.subplot(1,2,1)
	plt.plot(history.history['acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')

	plt.subplot(1,2,2)
	plt.plot(history.history['loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')

	plt.show()

## Pre-process training and test datasets
class_1, class_2 = 6,7 			# user selects two digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images: each image is reshaped to an 1x784 numpy array
x_train = np.reshape( x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[2]) )
x_test = np.reshape( x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]) )

# Keep only 'class_1' and 'class_2' digits from the dataset
X_train = np.vstack( (x_train[y_train==class_1], x_train[y_train==class_2]) )
Y_train = np.hstack( (y_train[y_train==class_1], y_train[y_train==class_2]) )
X_test = np.vstack( (x_test[y_test==class_1], x_test[y_test==class_2]) )
Y_test = np.hstack( (y_test[y_test==class_1], y_test[y_test==class_2]) )

# Shuffle training data and normalize both datasets
permute_indices = np.random.permutation(X_train.shape[0])
X_train = X_train[permute_indices,:]
Y_train = Y_train[permute_indices]

X_train = X_train/255.0
X_test = X_test/255.0

# Dìsplay an example train image
img_number = 18
plt.imshow(X_train[img_number,:].reshape(28,28), cmap="gray_r")
plt.title('Example of MNIST image - Image label: ' + str(Y_train[img_number]))
plt.axis('off')
plt.show()

# Binary (0,1) encoding of 'class_1' as 0 and 'class_2' as 1
Y_train[Y_train==class_1] = 0
Y_train[Y_train==class_2] = 1
Y_test[Y_test==class_1] = 0
Y_test[Y_test==class_2] = 1

## Build the model

# - Input layer: 784 nodes (since each image has 28x28 = 784 pixels)
# - Output layer: 1 node with sigmoid activation (output labeled as '0' or '1')
# - No hidden layers

np.random.seed(1)		# fix random seed for reproducibility

input_dim = X_train.shape[1]
output_dim = 1 							

model = Sequential()
model.add(Dense(output_dim, input_dim = input_dim, activation='sigmoid'))

# Compile  and train the model
sgd = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

num_epochs = 30
batch_size = 1024
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)

# Plot accuracy and error behavior
plot_metrics(history)

# Evaluate and predict
score = model.evaluate(X_test, Y_test, verbose=0) 
print('Test accuracy:', score[1])

Y_pred = model.predict_classes(X_test)

# Plot one example of prediction
img_number = 8
if Y_pred[img_number] == 0.0:
	predicted_class = class_1
else:
	predicted_class = class_2

plt.imshow(X_test[img_number,:].reshape(28,28), cmap="gray_r")
plt.title('Example of prediction - Image label: ' + str(predicted_class))
plt.axis('off')
plt.show()

# Classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix

target_names = ['class 1: ' + str(class_1), 'class 2: ' + str(class_2)]

print("Classification report:\n============================")
print(classification_report(Y_test, Y_pred,target_names=target_names))
print("Confusion matrix:\n============================")
print(confusion_matrix(Y_test, Y_pred))