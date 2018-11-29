from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

import math
import first_layer as fl
style.use('ggplot')


dim = 3
pixel = 255
pixel_half = 128
img_size = 28


# 根据feature的str，计算对应的向量组
def f_str_to_vec(_f_str):
    _mat = fl.str_1_to_mat(_f_str)
    # 这里反一下，与坐标一致
    (ind_y, ind_x) = np.nonzero(_mat)
    # TODO 检查坐标的长度一致
    _size = len(ind_x)
    if _size <= 4:
        _vec = [(ind_x[i],ind_y[i]) if i == 0 else (ind_x[i] - ind_x[i-1], ind_y[i] - ind_y[i-1]) for i in range(_size)]
    else:
        _vec = [0 for i in range(dim*dim - _size)]
        k = 0
        for i in range(dim):
            for j in range(dim):
                _v = _mat[i][j]
                if _v  == 0 :
                    # j,i 对应x和y坐标
                    _vec[k] = (j, i)
                    k += 1
        _vec = [(_vec[i][0], _vec[i][1]) if i == 0 else (_vec[i][0] - _vec[i - 1][0], _vec[i][1] - _vec[i - 1][1]) for i in
                range(len(_vec))]
    return _vec

# 计算vec的坐标,factor是初始位置的权重
def f_vec_to_coordi(_f_vec,factor = 0.3):
    _f_size = len(_f_vec)
    _x = 0
    _y = 0
    for i in range(_f_size):
        _v1_norm = math.sqrt(_f_vec[i][0] ** 2 + _f_vec[i][1] ** 2)
        if _v1_norm == 0:
            _v1_norm = 1
        if i == 0:
            _x += _f_vec[i][0] * factor / _v1_norm
            _y += _f_vec[i][1] * factor / _v1_norm
        else:
            _x += _f_vec[i][0] / _v1_norm
            _y += _f_vec[i][1] / _v1_norm
    return (_x,_y)

# 计算特征之间相似度
def f_dis(_fstr1,_fstr2):
    _v1 = f_str_to_vec(_fstr1)
    _v1_coordi = f_vec_to_coordi(_v1)
    _v2 = f_str_to_vec(_fstr2)
    _v2_coordi = f_vec_to_coordi(_v2)

    p = 1 - math.sqrt(((_v1_coordi[0] - _v2_coordi[0])**2 + (_v1_coordi[1] - _v2_coordi[1])**2)/8)
    return _v1,_v2,_v1_coordi,_v2_coordi,p

# 在分解平面展示特征
def f_show(dic,thresold = 0):
    dic_keys = list(dic.keys())
    dic_values = list(dic.values())
    size = len(dic_keys)

    dic_v_sum = sum(dic_values)
    dic_values = [dic_values[i] / dic_v_sum for i in range(size)]
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    x3 = [0 for i in range(size)]
    y3 = [0 for i in range(size)]
    z3 = np.zeros(size)

    dx = np.ones(size) * 0.1
    dy = np.ones(size) * 0.1
    dz = [0 for i in range(size)]

    # thresold = 0

    for i in range(len(dic_keys)):
        if i >= thresold:
            # (_x, _y) = f_deposi(f_str_to_vec(dic_keys[i]))
            (_x, _y) = f_vec_to_coordi(f_str_to_vec(dic_keys[i]))
            # x3[i], y3[i], dz[i] = _x, _y, dic_values[i]
            x3[i], y3[i], dz[i] = int(_x), int(_y), dic_values[i]

    ax1.bar3d(x3, y3, z3, dx, dy, dz)
    # x3, y3 = np.meshgrid(x3, y3)
    # ax1.plot_surface(x3, y3, dz, rstride=1, cstride=1, cmap='rainbow')

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.show()
    return

# dic = fl.load('../dict/test.txt')
dic = fl.load(fl.n_filename(1))
f_show(dic,3)

# f1 = '100100001'
# f2 = '100010001'
# v1,v2,v1_coordi,v2_coordi,p = f_dis(f1,f2)





