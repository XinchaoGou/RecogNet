import numpy as np
import cmath as cm
import json
import time
import os

from read_data import MinstData

dim = 3

# 返回矩阵的大小
# (n,m) = _size(img)
# n,m 分别为行数，列数
def _size(img):
    _image_size = np.array(img).shape
    n = _image_size[0]
    m = _image_size[1]
    return n,m

# 把图片分成len*len的小块,返回blocks
# blocks = split_block(img, 3)
# 返回3*3大小的分块，blocks[x][y]表示对应xy开始的3*3大小分块
def split_block(img, length):
    (n,m) = _size(img)

    blocks = [[[[img[x][y]
                for y in range(j,j+length)]
               for x in range(i,i+length) ]
              for j in range(m - length+1)]
             for i in range(n - length+1)]
    return blocks

# 计算ZNCC，返回的是绝对值
def zncc(block,pattern):
    n = m = dim
    _s_I = _s_I_2 = _s_P = _s_P_2 = _s_IP = 0.

    for i in range(n):
        for j in range(m):
            b_ij = block[i][j]
            # p_ij = 1 if pattern[i][j] else -1
            p_ij = pattern[i][j]
            _s_I += b_ij
            _s_I_2 += b_ij**2
            _s_P += p_ij
            _s_P_2 += p_ij**2
            _s_IP += b_ij * p_ij

    N = n*m
    d_I = N*_s_I_2 - _s_I**2
    d_J = N*_s_P_2 - _s_P**2
    if d_I == 0 and d_J != 0:
        d_I = d_J
    if d_I != 0 and d_J == 0:
        d_J = d_I
    if d_I == 0 and d_J == 0:
        return 1
    else:
        dem = (d_I)*(d_J)
    return abs((N*_s_IP - _s_I*_s_P)/cm.sqrt(dem))

# 生成模版 patterns = generate_patterns()
# 生成第1层的模版"二进制字符串"
# 不要更改_k 因为第2层的可能性就已经2417851639229258349412352种可能
def generate_patterns():
    _k = 1
    def _com_pattern(i):
        _p_str = bin(i)[2:]
        _total = (dim*dim) ** _k
        _current_len = len(_p_str)
        _com_str = "".join('0' for i in range(_total - _current_len))
        return _com_str + _p_str
    return [ _com_pattern(i) for i in range(2**((dim*dim)**_k -1))]

# 将二进字符串转化为对应矩阵，目前只转化最底层
# TODO 可能只是临时用 比如将'010010010' 解码为
# [[0, 1, 0],
#  [0, 1, 0],
#  [0, 1, 0]]
def str_1_to_mat(_str):
    _mats = [[int(_str[dim * i + j]) for j in range(dim)] for i in range(dim)]
    return _mats

# 将二进制字符串解码为字符矩阵
# 比如'010010010' 解码为
# [['0','1','0'],
#  ['0','1','0'],
#  ['0','1','0']]
def str_to_matstr(_str,_k):
    _len = len(_str)
    _block_n = dim * dim
    _block_size = _len//_block_n
    _mats = []
    if _k == 1:
        _mats = [[_str[dim * i + j] for j in range(dim)] for i in range(dim)]
    else:
        # 把二进制字符串分成9(dim * dim)块，迭代到低一层解码为字符矩阵
        _str_s = [ _str[ (k*_block_n) : (k*_block_n + _block_size)] for k in range(_block_n)]
        _mats = [[ str_to_matstr(_str_s[dim*i + j], _k -1) for j in range(dim)] for i in range(dim)]
    return _mats

# # 将第1层的矩阵模版，转换为十进制数字的字符串
# def mat_1_to_str(mat):
#     (n, m) = _size(mat)
#
#     _b_str = ''
#     for i in range(n):
#         for j in range(m):
#             _b_str += str(int(bin(mat[i][j]), 2))
#
#     _str =str(int(_b_str,2))
#     return _str



# 统计第一层数字n，模版的频次
# _num 是当前计算数字
# _patterns_str 是模版集的字符串形式
# _patterns 是模版集的矩阵形式
# mData 是训练样本的数据
# f_stop = 10 表示，新样本运算后，前10个频次最高的特征不变，则停止样本训练
def static_first_layer(_num, _patterns_str, _patterns, mData = MinstData(), f_stop = 10):
    # 模版频次统计字典
    p_dic = {}
    _pre_key_list = ['' for i in range(f_stop)]
    # 训练样本数
    _n_len = mData.get_length(_num)
    _file_name = '../dict/sort_dic_' + str(_num) +'.txt'

    for i in range( _n_len):
        img = mData.get_data(_num, i)
        p_dic = _single_image_dic(img, _patterns_str, _patterns, **p_dic)
        _t_dic_list = show_most_patterns(p_dic, f_stop)
        _t_stop_len = min(len(_t_dic_list), len(_pre_key_list), f_stop)
        _current_key_list = ['' for i in range(_t_stop_len)]

        _key = True
        for f_index in range(_t_stop_len):
            _current_key_list[f_index] = _t_dic_list[f_index][0]
            if _current_key_list[f_index] != _pre_key_list[f_index]:
                _key = False

        sort_dic = show_most_patterns(p_dic, f_stop, r_type = 'dict')

        print('数字'+ str(_num) + ': 前 '+ str(i+1) + ' 个样本累计频次')
        _save(sort_dic, _file_name)

        if not _key:
            _pre_key_list = _current_key_list[:]
        else:
            print('计算了' + str(i+1) + '个样本\n')
            return p_dic

    return p_dic

# 按频次降序排列，show=True 时，打印出现频次最多的n个模版
def show_most_patterns(p_dic, n = 10, r_type = 'list',show=False):
    sort_dic = sorted(p_dic.items(), key=lambda item: item[1], reverse=True)
    if show :
        for i in range(n):
            _mat = str_1_to_mat(sort_dic[i][0])
            print(sort_dic[i][0] + '\t频次\t' + str(sort_dic[i][1]))
            for j in range(3):
                print(_mat[j])
            print('\n')

    # 默认返回list，按需返回dict
    if r_type =='dict':
        return _list_to_dict(sort_dic)

    return sort_dic

# 保存频次统计的数据到文件
def _save(m_dic, _file_name = '../dict/sort_dic.txt'):
    with open(_file_name, 'w') as openfile:
        json.dump(m_dic, openfile)
        print('已更新文件'+_file_name+'！')
    return

# 加载 频次统计文件 到 字典数据
def _load(_file_name = '../dict/sort_dic.txt'):
    if os.path.exists(_file_name):
        with open(_file_name, 'rb') as loadfile:
            load_dic = json.load(loadfile)
            print('已加载文件'+_file_name+'！')
    else:
        print('加载' + _file_name + '失败！文件不存在！')
        load_dic = {}
    return load_dic

# 增加当前频次统计字典的数据 到 总的频次统计数据
def _cDict_to_allDict(m_dic, _all_file_name = '../dict/all_frequent_dict.txt'):

    _all_frequent_dict = _load(_all_file_name)

    for key, value in m_dic.items():
        _all_frequent_dict[key]= _all_frequent_dict.get(key,0) + value

    # 保存新的总频次数据
    _save(_all_frequent_dict, _all_file_name)
    return

# 频次统计的list转换为dict
def _list_to_dict(_list):
    return {_list[i][0]: _list[i][1] for i in range(len(_list))}

# 传入图像 _img ，统计图像上各特征集 _patterns 中特征的频次字典
# 如果有 _pre_dic 字段 则将 该样本频次字典 与 历史频次字典 合并
# _pattern_strs 是特征 为字符串 形式的特征集，list类型
# _patterns 是 特征 为矩阵 形式的特征集，list类型
# 返回值是dic类型
def _single_image_dic(_img, _pattern_strs, _patterns, **_pre_dic):
    blocks = split_block(_img, dim)
    n, m = _size(blocks)
    # 特征频次统计字典,有则在原来基础上添加，无则输出当前图的统计字典
    if _pre_dic is not None:
        p_dic = _pre_dic
    else:
        p_dic = {}

    for n_index in range(n):
        for m_index in range(m):
            p_max = 0
            p_max_index = ''
            _block = blocks[n_index][m_index]

            for p_index in range(len(_patterns)):
                _pattern = _patterns[p_index]
                p = zncc(_block, _pattern)
                if p >= p_max:
                    p_max = p
                    p_max_index = p_index

            # 也可以用矩阵形式，或者字符串形式互相转换，但是考虑这样在大量调用时，计算量太大
            # 所以这里直接索引来得到
            _max_pattern_str = _pattern_strs[p_max_index]
            # dit.get 优化
            p_dic[_max_pattern_str] = p_dic.get(_max_pattern_str, 0) + 1
    return p_dic

# 运行训练样本数据，生成频次字典
# f_stop_num = 50 设定前50个特征渐进稳定的时候，停止当前数字训练，用于加速
def _run_train_data(f_stop_num = 50):
    start = time.time()
    # 生成模版
    patterns_str = generate_patterns()
    patterns = [str_1_to_mat(patterns_str[i]) for i in range(len(patterns_str))]
    for i in range(2):
        p_dic = static_first_layer(i, patterns_str, patterns, f_stop=f_stop_num)
        sort_dic = show_most_patterns(p_dic, f_stop_num, 'dict', True)
        _cDict_to_allDict(sort_dic)

    print('运行时间' + str(time.time() - start))
    return

_run_train_data(5)









