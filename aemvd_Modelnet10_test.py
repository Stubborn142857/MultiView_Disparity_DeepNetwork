from MVD1 import AEMVD as aemvd
from PIL import Image
import numpy as np
import os, fnmatch
import time
import keras
import matplotlib.pyplot as plt

_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def length_of_dataset(path, type):
    class_list = os.listdir(path)
    i = 0
    for c in class_list:
        obj_dir_path = path + '/' + c + '/' + type
        # print (obj_dir_path)
        obj_list = os.listdir(obj_dir_path)
        i += len(obj_list)
    print (i)
    return i


def load_dataset(path, type):
    obj_num = length_of_dataset(path, type)
    x = np.zeros((obj_num, 128, 128, 12))
    y = np.zeros((obj_num, 64, 64, 12))
    # first = True
    class_list = os.listdir(path)
    i = 0
    for c in class_list:
        obj_dir_path = path + '/' + c + '/' + type
        # print (obj_dir_path)
        obj_list = os.listdir(obj_dir_path)
        for obj in obj_list:
            obj_view_path = obj_dir_path + '/' + obj
            # print (obj_view_path)
            # file_list = fnmatch.filter(os.listdir(obj_view_path), 'view_*')
            for v_i in range(0, 12):
                x_file_path = obj_view_path + '/view_' + str(v_i) + '.png'
                y_file_path = obj_view_path + '/depth_' + str(v_i) + '.png'
                # print (file_path)
                img = Image.open(x_file_path).convert('L')
                img.load()
                img = img.resize((x.shape[2], x.shape[1]), Image.ANTIALIAS)
                data = np.asarray(img, dtype="int32")
                x[i, :, :, v_i] = data

                img = Image.open(y_file_path).convert('L')
                img.load()
                img = img.resize((y.shape[2], y.shape[1]), Image.ANTIALIAS)
                data = np.asarray(img, dtype="int32")
                y[i, :, :, v_i] = data

            i += 1
            print(i)

    x[x >= 255] = 0
    return x, y


def normal(X):
    min = np.min(np.min(X))
    max = np.max(np.max(X))
    X = (X - min) / max
    return X


# Initialize Dataset
# path = "E:\Dataset\ModelNet10\ModelNet10_multiview"
# x_train, y_train = load_dataset(path, 'train')
# x_test, y_test = load_dataset(path, 'test')
# x_train = normal(x_train)
# x_test = normal(x_test)
# y_train = normal(y_train)
# y_test = normal(y_test)
#
# np.save('x_train.npy', x_train)
# np.save('x_test.npy', x_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)

# x_train=np.load('x_train.npy')
# print('x train loaded')
x_test = np.load('x_test.npy')
print('x test loaded')

# y_train=np.load('y_train.npy')
# print('y train loaded')
y_test = np.load('y_test.npy')
print('y test loaded')

# load the aemvd network
model_name = 'model_aemvd_modelnet10_1'

model = aemvd.load_model_and_weight('model/' + model_name)
model = aemvd.compile(model)

x=x_test[30:31,:,:,:]
p = model.predict(x, batch_size=aemvd.get_batch_size(), verbose=0)
img = y_test[30, :, :, 0]
depth = p[0, :, :, 0]
# img = img * 255
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(img, cmap="hot")
ax2.imshow(depth, cmap="hot")
plt.show()
