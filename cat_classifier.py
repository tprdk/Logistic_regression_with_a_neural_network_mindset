import h5py
from PIL import Image
import numpy as np
from Logistic_Regression import Logistic_Regression

#file path parameters
TRAIN_FILE_PATH = 'dataset\\train_catvnoncat.h5'
TEST_FILE_PATH  = 'dataset\\test_catvnoncat.h5'

#train params
EPOCHS          = 20000
LEARNING_RATE   = 0.005


#train file includes ['list_classes', 'train_set_x', 'train_set_y']
#test  file includes ['list_classes', 'test_set_x',  'test_set_y']
def load_data_from_h5_file(file_path):
    with h5py.File(file_path, "r") as f:
        x_tag = list(f.keys())[1]
        y_tag = list(f.keys())[2]
        return np.array(list(f[x_tag]), dtype='float32'), np.array(list(f[y_tag]), dtype='float32')

#print function for images
def print_images(images):
    for image in images:
        im = Image.fromarray(image)
        im.show()


#read test and train data
x_train, y_train = load_data_from_h5_file(TRAIN_FILE_PATH)
x_test,  y_test  = load_data_from_h5_file(TEST_FILE_PATH)

#flatten images
x_test = x_test.reshape(x_test.shape[0], -1).T
x_train = x_train.reshape(x_train.shape[0], -1).T

#normalize the images
x_test  = x_test  / 255.0
x_train = x_train / 255.0

#init log_reg model
model = Logistic_Regression(EPOCHS, LEARNING_RATE)

model.train_model(x_train, y_train)
model.test_model(x_train, y_train)
model.test_model(x_test, y_test)

