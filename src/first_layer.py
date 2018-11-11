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
# f_stop = 10 表示，新样本运算后，前10个频次最高的特征不变，则停止样本训练
def static_first_layer(_num , f_stop = 10):
    # 生成模版
    patterns_str = generate_patterns()
    patterns = [str_1_to_mat(patterns_str[i]) for i in range(len(patterns_str))]
    # 读取数据
    mData = MinstData()
    # 模版频次统计字典
    p_dic = {}
    _pre_key_list = ['' for i in range(f_stop)]
    # 训练样本数
    # _n_len = 100
    _n_len = mData.get_length(_num)

    for i in range( _n_len):
        img = mData.get_data(_num, i)
        blocks = split_block(img, dim)
        n, m = _size(blocks)

        for n_index in range(n):
            for m_index in range(m):
                p_max = 0
                p_max_index = ''
                _block = blocks[n_index][m_index]

                for p_index in range(len(patterns)):
                    _pattern = patterns[p_index]
                    p = zncc(_block, _pattern)
                    if p >= p_max:
                        p_max = p
                        p_max_index = p_index

                _max_pattern_str = patterns_str[p_max_index]
                # dit.get 优化
                p_dic[_max_pattern_str] = p_dic.get(_max_pattern_str,0) +1

        # 累计前i个样本，按频次降序排列的特征
        _t_dic_list = show_most_patterns(p_dic, f_stop)

        # 累计前 i 个样本，出现过的 不同特征 的总数
        _t_stop_len = min(len(_t_dic_list), f_stop, len(_pre_key_list))

        _current_key_list = ['' for i in range(_t_stop_len)]

        # 比较累计 i 个的频次统计与累计 i-1 个样本的频次统计
        # 频次最高的，前 _t_stop_len 个特征，是否完全一致

        _key = True
        for f_index in range(_t_stop_len):
            _current_key_list[f_index] = _t_dic_list[f_index][0]
            if _current_key_list[f_index] != _pre_key_list[f_index]:
                _key = False
        # 更新累计统计模版
        # p_dic已经是前 i 个样本的累计频次，直接覆盖即可
        sort_dic = show_most_patterns(p_dic, f_stop, r_type = 'dict')
        print('前个'+ str(i+1) +'样本累计频次')
        _save(sort_dic)

        # 如果前 f_stop 个特征完全相同，则停止训练
        if not _key:
            _pre_key_list = _current_key_list[:]
        else:
            print('计算了' + str(i) + '个样本\n')
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
        print('已存储到文件'+_file_name+'！')
    return

# 加载频次统计文件到数据
def _load(_file_name = '../dict/sort_dic.txt'):
    with open(_file_name, 'rb') as loadfile:
        load_dic = json.load(loadfile)
        print('已加载文件'+_file_name+'！')
    return load_dic

# 增加当前频次统计字典的数据 到 总的频次统计数据
def _cDict_to_allDict(m_dic, _all_file_name = '../dict/all_frequent_dict.txt'):

    _all_frequent_dict = {}
    if os.path.exists(_all_file_name):
        _all_frequent_dict = _load(_all_file_name)

    for key, value in m_dic.items():
        _all_frequent_dict[key]= _all_frequent_dict.get(key,0) + value

    # 保存新的总频次数据
    _save(_all_frequent_dict, _all_file_name)
    return

# 频次统计的list转换为dict
# TODO 也许可以换成lambda
def _list_to_dict(_list):
    return {_list[i][0]: _list[i][1] for i in range(len(_list))}


start = time.time()

f_stop_num = 100
p_dic = static_first_layer(1, f_stop_num)
sort_dic = show_most_patterns(p_dic, f_stop_num, 'dict', True)
_save(sort_dic)
# my_dic = _load()
_cDict_to_allDict(sort_dic)

end = time.time()
print('运行时间' + str(end - start))






