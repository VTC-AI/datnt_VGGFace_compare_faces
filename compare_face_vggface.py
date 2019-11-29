from keras.models import model_from_json, Sequential, Model
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np


model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# download vgg_face_weights.h below link
# https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view
model.load_weights('train/vgg_face_weights.h5')

vgg_face_descriptor = Model(inputs=model.model.layers[0].input, outputs=model.layers[-2].output)

def preprocess_image(image_path):
  img = load_img(image_path, target_size=(224, 224))
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_input(img)
  return img

# img1_present = vgg_face_descriptor.predict(preprocess_image('1.jpg'))[0, :]

def find_cosin_distance(source_represent, test_represent):
  a = np.matmul(np.transpose(source_represent), test_represent)
  b = np.sum(np.multiply(source_represent, source_represent))
  c = np.sum(np.multiply(test_represent, test_represent))
  return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def find_euclidean_distance(source_represent, test_represent):
  e_d = source_represent - test_represent
  e_d = np.sum(np.multiply(e_d, e_d))
  e_d = np.sqrt(e_d)
  return e_d


epsilon = 0.40
def verify_face(img1, img2):
  img1_represent = vgg_face_descriptor.predict(preprocess_image(img1))[0, :]
  img2_represent = vgg_face_descriptor.predict(preprocess_image(img2))[0, :]

  similar = find_cosin_distance(img1_represent, img2_represent)
  e_d = find_euclidean_distance(img1_represent, img2_represent)
  print(similar)
  print(e_d)

  if similar < epsilon:
    print('Same person')
  else:
    print('Not same person')

verify_face('8.jpg', '7.jpeg')
