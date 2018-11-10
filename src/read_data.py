import os
import struct
import numpy as np
# import matplotlib.pyplot as plt

"""
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
"""


def load_mnist(path='../trainData', kind='train'):
    """从给定路径读取minst数据"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        """以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式"""
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        """一共读8个字节，struct把读到的字符串依次变成两个对应的4字节无符号整数"""
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
        """fromfile()函数按照无符号单字节整数来读取数据"""

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                              imgpath.read(16))
        """以上每个对应4个字节"""
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        """按照无符号单字节整数来读取灰度值"""
    return images, labels


class MinstData(object):

    x_train, y_train = load_mnist('../trainData', kind='train')

    def get_data(self, _num, _index):
        data = self.x_train[self.y_train == _num][_index].reshape(28, 28)
        return data

    def get_length(self, _num):
        length = len(self.x_train[self.y_train == _num])
        return length

# a=MinstData()
# a_data=a.get_data(5, 255)
# length=a.get_length(6)

# fig, ax = plt.subplots(
#     nrows=2,
#     ncols=5,
#     sharex=True,
#     sharey=True, )
#
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()