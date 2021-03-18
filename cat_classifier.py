import numpy as np
import h5py

TRAIN_FILE_PATH = 'dataset\\train_catvnoncat.h5'
TEST_FILE_PATH  = 'dataset\\test_catvnoncat.h5'

#train file includes ['list_classes', 'train_set_x', 'train_set_y']
#test  file includes ['list_classes', 'test_set_x',  'test_set_y']
def load_data_from_h5_file(file_path):
    with h5py.File(file_path, "r") as f:
        print("Keys: %s" % f.keys())
        x_tag = list(f.keys())[1]
        y_tag = list(f.keys())[2]
        return list(f[x_tag]), list(f[y_tag])


x_train, y_train = load_data_from_h5_file(TRAIN_FILE_PATH)
x_test,  y_test  = load_data_from_h5_file(TEST_FILE_PATH)



