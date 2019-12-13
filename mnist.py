from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

print(train_images.shape)
print(len(train_labels))


# the network architecture/ layes adding

from keras import models
from keras import layers

network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

# compilation step

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# preparing the image data
train_images=train_images.reshape((60000,28*28))
#print(train_images)

train_images=train_images.astype('float32')/255
print(train_images)

# preparing test images
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# preaparing the test_labels

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)


# train model

network.fit(train_images,train_labels,epochs=10,batch_size=128)


# print test Accuracy
test_loss,test_acc=network.evaluate(test_images,test_labels)
print("Test Accuracy: ",test_acc)

# print(test_images.item(1))
