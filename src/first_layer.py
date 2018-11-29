import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cmath as cm
import json
import time
import random
import math
import copy
import os

from PIL import Image
from read_data import MinstData



dim = 3
pixel = 255
pixel_half = 128
img_size = 28

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
    d_block = N*_s_I_2 - _s_I**2
    d_Pattern = N*_s_P_2 - _s_P**2
    if d_block == 0 and d_Pattern != 0:
        d_block = d_Pattern
    if d_block != 0 and d_Pattern == 0:
        d_Pattern = d_block
    if d_block == 0 and d_Pattern == 0:
        # 均值低于一半则为0，高于一半则为1
        _threshold = N * pixel_half
        return 1 if ((_s_P == 0) and (_s_I< _threshold)) or ((_s_P > 0) and (_s_I > _threshold)) else 0
    else:
        dem = (d_block)*(d_Pattern)
    return (N*_s_IP - _s_I*_s_P)/cm.sqrt(dem)

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
    return [ _com_pattern(i) for i in range(2**((dim*dim)**_k))]

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
# 这样递归的设计可以便于分布式计算，所以没有直接进行二进制字符串的比较
# 而且字串相同的可以直接返回1，不用计算所有，便于加速
# 因为每一层建立再有限个低层模版上，作用更强
# 有一个字符串则返回字符矩阵
# 有两个字符串则比较相似度
def str_to_matstr_or_compare(_str, _str_2=None):
    _len = len(_str)
    _k = _len// (dim * dim)
    # TODO 检查字符串长度是否符合要求 _len % (dim * dim) == 0
    # TODO 如果两个模版的长度不同（不在同一层），可以用其他办法比较
    _block_n = dim * dim
    _block_size = _len//_block_n
    _mats = []
    if _k == 1:
        _mats = [[_str[dim * i + j] for j in range(dim)] for i in range(dim)]
        if _str_2 is not None:
            _mats_2 = str_to_matstr_or_compare(_str_2)
            _not_xor = lambda x,y: 1 if x == y else 0
            _xor_mat = [[_not_xor(_mats[i][j],_mats_2[i][j]) for j in range(dim)] for i in range(dim)]
            return sum(map(sum, _xor_mat))/(dim * dim)
    else:
        # 把二进制字符串分成9(dim * dim)块，迭代到低一层解码为字符矩阵
        _str_s = [ _str[ (k*_block_n) : (k*_block_n + _block_size)] for k in range(_block_n)]
        _mats = [[str_to_matstr_or_compare(_str_s[dim * i + j]) for j in range(dim)] for i in range(dim)]
        if _str_2 is not None:
            if _str == _str_2:
                return 1
            _mats_2 = str_to_matstr_or_compare(_str_2)
            _str_s_2 = [ _str_2[ (k*_block_n) : (k*_block_n + _block_size)] for k in range(_block_n)]
            _xor_mat = [[str_to_matstr_or_compare(_str_s[dim * i + j], _str_s_2[dim * i + j]) for j in range(dim)] for i in range(dim)]
            return sum(map(sum, _xor_mat)) / (dim * dim)

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
        p_dic = _single_image_dic_or_real_patterns(img, _patterns_str, _patterns, **p_dic)
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
def load(_file_name ='../dict/all_frequent_dict.txt'):
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

    _all_frequent_dict = load(_all_file_name)

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
# 返回值是dic类型,
# 如果 output_real_patterns = True 则返回
# 图像在第一组基下的 第一层网络 (26*26) 大小的list 和 每个神经元的概率（输出能量大小）
def _single_image_dic_or_real_patterns(_img, _pattern_strs, _patterns, output_real_patterns = False, **_pre_dic):
    blocks = split_block(_img, dim)
    n, m = _size(blocks)
    # 特征频次统计字典,有则在原来基础上添加，无则输出当前图的统计字典
    # 如果没参数，系统自动分配了 _pre_dic = {}
    p_dic = _pre_dic

    if output_real_patterns:
        _blocks_f_strs = [ [ '' for j in range(m)] for i in range(n)]
        _blocks_f_confidence = [[ 0. for j in range(m)] for i in range(n)]

    for n_index in range(n):
        for m_index in range(m):
            p_max = 0.
            p_max_index = ''
            _block = blocks[n_index][m_index]

            for p_index in range(len(_patterns)):
                _pattern = _patterns[p_index]
                p = zncc(_block, _pattern)
                if p >= p_max:
                    p_max = p
                    p_max_index = p_index

            _max_pattern_str = _pattern_strs[p_max_index]
            if output_real_patterns:
                _blocks_f_strs[n_index][m_index] = _max_pattern_str
                _blocks_f_confidence[n_index][m_index]  = p_max
            # dit.get 优化
            p_dic[_max_pattern_str] = p_dic.get(_max_pattern_str, 0) + 1
    if output_real_patterns:
        return _blocks_f_strs,_blocks_f_confidence
    return p_dic

# 运行训练样本数据，生成频次字典
# f_stop_num = 50 设定前50个特征渐进稳定的时候，停止当前数字训练，用于加速
def _run_train_data(f_stop_num = 50):
    start = time.time()
    # 生成模版
    patterns_str = generate_patterns()
    patterns = [str_1_to_mat(patterns_str[i]) for i in range(len(patterns_str))]
    for i in range(10):
        p_dic = static_first_layer(i, patterns_str, patterns, f_stop=f_stop_num)
        sort_dic = show_most_patterns(p_dic, f_stop_num, 'dict', True)
        _cDict_to_allDict(sort_dic)

    print('运行时间' + str(time.time() - start))
    return

# 根据给定数字，生成对应文件路径
def n_filename(_num):
    return '../dict/sort_dic_' + str(_num) +'.txt'

# 从特征的字符串找出特征编号
def fstr_to_index(_feature_str):
    _str = '0b'+_feature_str
    return int(_str,2)

# 只用计算一半,不是矩阵形式，第一行 512 个，第二行 511 个，很行依次递减一
# 计算全部 1 层特征之间的相似矩阵,大小为 512 * i
# _all_f_mat[i][j]表示 特征i 与 特征j 的相似度
# 返回 相似度的半矩阵
# TODO 默认计算所有特征相似度，也可以只计算传入的特征基之间的相似度矩阵,
# 用的时候注意需要给 这组 传入的特征基建立一个索引的字典key是特征，value是对应相似矩阵的角标
def _pattern_str_distance_mat(_patterns_str = generate_patterns()):
    _all_feature_num = len( _patterns_str)
    _all_f_mat = [[str_to_matstr_or_compare(_patterns_str[i], _patterns_str[j]) for j in range(i, _all_feature_num)] for i in range(_all_feature_num)]
    return _all_f_mat

# 查表两个特征之间距离,因为矩阵只有一半注意索引
# 调用时 dis_from_mat(_str_1,_str_2,_all_f_mat) 计算全部特征集上的特征之间距离
# 调用dis_from_mat(_str_1,_str_2,_all_f_mat, **dic)时，
# 特征集是一部分，所以必须建立特征字符串 和到相似度矩阵对应索引 的字典
def dis_from_mat(_str_1,_str_2,_all_f_mat, ** _dic):

    if _dic == {}:
        ind_1 = fstr_to_index(_str_1)
        ind_2 = fstr_to_index(_str_2)
    else:
        ind_1 = _dic[_str_1]
        ind_2 = _dic[_str_2]

    if ind_1 < ind_2:
        _min_ind, _max_ind = ind_1, ind_2
    else:
        _min_ind, _max_ind = ind_2, ind_1
    return _all_f_mat[_min_ind][_max_ind - _min_ind]

# 将给定的特征，映射到给定的基中, 需要传入对应的 特征相似半矩阵进行查找
# 返回映射到的特征的str，同时返回该映射的操作的置信率
def _update_feature(_fstr, _base_features_strs, _dis_mat):

    _p_list = [ dis_from_mat(_fstr, _base_features_strs[i], _dis_mat) for i in range(len(_base_features_strs))]
    _max_confidence = max(_p_list)
    _index = _p_list.index(_max_confidence)
    return _base_features_strs[_index], _max_confidence

# 传入该层网络，传入一组特征基，传入特征相似半矩阵，用于查找特征相似度
def _updat_layer(_layer, _base_features, _dis_mat):
    _size = len(_layer)
    _updated_layer = [['' for j in range(_size)] for i in range(_size)]
    _updated_layer_conf = [[0. for j in range(_size)] for i in range(_size)]
    for i in range(_size):
        for j in range(_size):
            _new_feature_str, _max_confidence = _update_feature(_layer[i][j], _base_features, _dis_mat)
            _updated_layer[i][j] = _new_feature_str
            _updated_layer_conf[i][j] = _max_confidence

    return _updated_layer, _updated_layer_conf

# 根据网络层的输出以及每个神经元的置信概率，重建图像
def _layer_to_img(_layer,_layer_p):
    _re_img = [[0 for j in range(img_size)] for i in range(img_size)]
    _size = len(_layer)
    for i in range( _size):
        for j in range(_size):
            _pattern_str = _layer[i][j]
            _pattern_probablity = _layer_p[i][j]
            _pattern = str_1_to_mat(_pattern_str)
            for n_index in range(dim):
                for m_index in range(dim):
                    # TODO 这里可以再结合每个特征的生成误差,特征映射误差,以及特征的频次比,为权重
                    _re_img[i][j] += _pattern[n_index][m_index] * _pattern_probablity
    return _re_img

# 归一化图像
def normalize(_img):
    max_value = max(max(_img))
    return [[_img[i][j]*pixel/max_value for j in range(img_size)] for i in range(img_size)]

# 第一层修正，根据给定的基，生成修正图像
# 根据前 _f_num 个特征为基
# 数字 _num 如果给定用对应的特征，否则用全部训练的特征
# TODO 未完成
def _show_img(_img, _f_num = 10, _num = None):
    mData = MinstData()
    num = 1
    f_num = 100
    img = mData.get_data(num, 0)
    # 生成模版
    patterns_str = generate_patterns()
    patterns = [str_1_to_mat(patterns_str[i]) for i in range(len(patterns_str))]
    dis_mat = _pattern_str_distance_mat(patterns_str)
    # 第一层的实际模版,及对应概率
    layer_1, Layer_1_p = _single_image_dic_or_real_patterns(img, patterns_str, patterns, output_real_patterns=True)

    best_features_dic = load(n_filename(num))
    best_features = [key for key in best_features_dic][0:f_num]

    # 更新第一层
    updated_layer,updated_layer_conf = _updat_layer(layer_1, best_features, dis_mat)

    re_img = _layer_to_img(layer_1, Layer_1_p)

    # 直接归一化的效果应该更好，可以反应每层神经网络对原始信息扭曲后的结果
    n_img = normalize(re_img)
    b_img = [[1 if re_img[i][j] > 4 else 0 for j in range(img_size)] for i in range(img_size)]
    b_img = normalize(b_img)

    show_img = n_img

    for i in range(img_size):
        for j in range(img_size):
            img[i][j] = show_img[i][j]

    Image.fromarray(img).show()
    return img

# 根据样本，第一层网络的输出，计算频次统计图，与学习到的对应标签的频次统计图计算相似度
# _dic 是降序排列的频次字典
# _f_num 是前n个特征基
# _pattern_strs 和 _patterns 是所有特征的
def _p_img_to_tag(_img,_f_num, _pattern_strs, _patterns, _dic, dis_mat = _pattern_str_distance_mat()):
    # 归一化前几个特征的频次
    def normalize_value(_value_list):
        _sum = sum(_value_list)
        return [_value_list[i]/_sum for i in range(len(_value_list))]

    # 计算单个样本的第一层网络实际输出
    _sample_layer_1,_sample_layer_1_conf = _single_image_dic_or_real_patterns(_img, _pattern_strs, _patterns, output_real_patterns = True)

    best_features = list(_dic.keys())[0:_f_num]
    best_features_value = normalize_value(list(_dic.values())[0:_f_num])

    _updated_layer, _updated_layer_conf = _updat_layer(_sample_layer_1, best_features, dis_mat)
    # 根据每个区块实际投影误差，区块特征矫正误差，生成矫正后频次图
    _updated_layer_dic ={}
    _size = len(_updated_layer)
    for i in range(_size):
        for j in range(_size):
            _key =  _updated_layer[i][j]
            _real_project_confidence =  _sample_layer_1_conf[i][j]
            _change_feature_confidence = _updated_layer_conf[i][j]
            _updated_layer_dic[_key] = _updated_layer_dic.get(_key,0) + _real_project_confidence * _change_feature_confidence

    # 矫正后的频次图和对应标签学习到的频次图相似度计算
    # 归一化频次图，用公式算概率
    _sample_features = list(_updated_layer_dic.keys())[0:_f_num]
    _sample_features_value = normalize_value(list(_updated_layer_dic.values())[0:_f_num])
    for i in range(len(_sample_features)):
        _updated_layer_dic[_sample_features[i]] = _sample_features_value[i]

    _similarity = 0.
    for i in range(len(best_features)):
        _best_key = best_features[i]
        _p_beast = best_features_value[i]
        _p_sample = _updated_layer_dic.get(_best_key,0)
        # _similarity +=_p_beast * math.exp( -abs(_p_beast - _p_sample) )
        _similarity += (_p_beast - _p_sample) ** 2

    return np.sqrt(_similarity)

# 输出分类标签
def _classification(_img,_f_num,_patterns_str,_patterns,best_dics_list = None ):
    if best_dics_list is not None:
        pass
    else:
        best_dics_list = [load(n_filename(i)) for i in range(10)]
    _tag_list = [0 for i in range(10)]
    dis_mat = _pattern_str_distance_mat(_patterns_str)
    for i in range(10):
        best_features_dic = best_dics_list[i]
        _p = _p_img_to_tag(_img, _f_num, _patterns_str, _patterns, best_features_dic,dis_mat)
        _tag_list[i] = _p
    return _tag_list.index(min(_tag_list)), _tag_list

# TODO 未完成 测试分类器对数字 num 成功率
def test(_num):
    mData = MinstData()
    num = _num
    f_num = 10
    # 生成模版
    patterns_str = generate_patterns()
    patterns = [str_1_to_mat(patterns_str[i]) for i in range(len(patterns_str))]
    # 读取认知图,特征聚类后作为字典
    best_dic_list = [_features_cluster(load(n_filename(i)), f_num) for i in range(10)]
    start = time.time()

    success = 0
    fail = 0
    total = 0
    for j in range(450, 500, 5):
        total += 1
        img = mData.get_data(num, j)
        tag, _tag_list = _classification(img,f_num, patterns_str, patterns, best_dics_list=best_dic_list)
        print('运行时间' + str(time.time() - start))
        print('分类器输出为' + str(tag))
        if tag == num:
            success += 1
            _p_real_tag_dis = _tag_list[0]
            _p_max_dis = max(_tag_list)
            print('成功率' + str(success / total) + '\t理论置信率 ' + str(_p_real_tag_dis/_p_max_dis))
        else:
            fail += 1
            _p_real_tag_dis = _tag_list[0]
            _p_output_tag_dis = _tag_list[tag]
            _p_max_dis = max(_tag_list)
            _p_confidence = abs(_p_real_tag_dis - _p_output_tag_dis) / _p_max_dis
            print('失败率' + str(fail / total) + '\t理论置信率 ' + str(1 - _p_confidence))
    return

# 特征聚类,聚为 _k 类
def _features_cluster(_features_dic, _k_class):
    def normalize_value(_value_list):
        _sum = sum(_value_list)
        return [_value_list[i]/_sum for i in range(len(_value_list))]

    # 将给定特征和其权重累加到总和里,输入输出都是list
    def sum_features(_fmat_0,_fmat_1):
        _mat_0 = np.array(_fmat_0)
        _mat_1 = np.array(_fmat_1)
        return (_mat_0+_mat_1).tolist()

    # 将mat归一化，根据给定归一化的模版，生成对应的字符串
    def mat_to_fstr(_mat, _threshold):
        _str = ''
        for i in range(dim):
            for j in range(dim):
                _mat[i][j]
                if _mat[i][j] >= _threshold/2:
                    _str += '1'
                else:
                    _str += '0'
        return _str

    # 将给入的 特征基 收敛到给定的 种子
    def _singel_stage(_str,_val):
        # 用来存储 每个样本的标签对应特征种子的索引，是 _str 中的index
        _nearest_k = [0 for i in range(_features_len)]

        for i in range(len(_features_strs)):
            _p_list = [0. for k in range(_k_class)]
            for _k in range(_k_class):
                _k_feature = _str[_k]
                _feature = _features_strs[i]
                _p_list[_k] = dis_from_mat(_k_feature, _feature, dis_mat)
            # 找到该特征对应的种子
            _nearest_k[i] = _p_list.index(max(_p_list))

        # 下一步所有种子相同的，计算加权平均值，生成特征和对应权重，生成新的字典
        for _k in range(_k_class):
            _k_mean_mat = [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]
            _k_num = 0
            _k_v = 0
            # 遍历所有特征对应的种子标签是 _k 的特征
            for i in range(len(_nearest_k)):
                # 如果特征对应的种子标签是 _k
                if _nearest_k[i] == _k:
                    _feature = _features_strs[i]
                    _feature_value = _features_value[i]
                    _k_mean_mat = sum_features(_k_mean_mat, str_1_to_mat(_feature))
                    _k_v += _feature_value
                    _k_num += 1
            _str[_k] = mat_to_fstr(_k_mean_mat, _k_num)
            _val[_k] = _k_v
        return _str, _val

    # 根据给定的key和value的列表，生成字典
    def _key_value_to_new_dic(_key_list,_value_list):
        _dic = {}
        # TODO 长度检查
        _len = len(_key_list)
        for i in range(_len):
            _f = _key_list[i]
            _v = _value_list[i]
            _dic[_f] = _v
        return _dic

    _features_strs = list(_features_dic.keys())
    _features_value = normalize_value(list(_features_dic.values()))
    _features_len = len(_features_strs)
    # 所有特征的相似矩阵
    dis_mat = _pattern_str_distance_mat()

    # 初始化 _k 个初始点,random 包含首位，所以减一！
    _k_str = [ _features_strs[random.randint(0, _features_len - 1)] for i in range(_k_class)]
    _k_val = [ 0. for i in range(_k_class)]
    new_k_str,new_k_val = _singel_stage(_k_str, _k_val)
    _k_str = copy.deepcopy(new_k_str)
    _k_val = copy.deepcopy(new_k_val)
    # _k_str,_k_val=new_k_str,new_k_val
    for i in range(100):
        _k_str, _k_val = _singel_stage(_k_str, _k_val)
        if new_k_str == _k_str:
            break
        else:
            new_k_str =copy.deepcopy(_k_str)
            new_k_val =copy.deepcopy(_k_val)
    return _key_value_to_new_dic(_k_str, _k_val)

def _history():
    # _run_train_data(100)

    # test(1)

    mData = MinstData()
    num = 7
    f_num = 10
    # 生成模版
    patterns_str = generate_patterns()
    patterns = [str_1_to_mat(patterns_str[i]) for i in range(len(patterns_str))]
    # 读取认知图,特征聚类后作为字典
    # p1 = {key: value for key, value in prices.items() if value > 200}

    best_dic_list = [_features_cluster(load(n_filename(i)), f_num) for i in range(10)]
    start = time.time()

    success = 0
    fail = 0
    total = 0
    for j in range(450, 500, 5):
        total += 1
        img = mData.get_data(num, j)
        tag, _tag_list = _classification(img, f_num, patterns_str, patterns, best_dics_list=best_dic_list)
        print('运行时间' + str(time.time() - start))
        print('分类器输出为' + str(tag))
        if tag == num:
            success += 1
            print('成功率' + str(success / total))
        else:
            fail += 1
            _p_real_tag_dis = _tag_list[num]
            _p_output_tag_dis = _tag_list[tag]
            _p_max_dis = max(_tag_list)
            _p_confidence = abs(_p_real_tag_dis - _p_output_tag_dis) / _p_max_dis
            print('失败率' + str(fail / total) + '\t理论置信率 ' + str(1 - _p_confidence))
    return



