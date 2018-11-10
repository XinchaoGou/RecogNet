import numpy as np
import cmath as cm
import sys

# img = [[0, 0, 1, 0, 0],
#        [0, 1, 0, 1, 0],
#        [1, 0, 0, 0, 1],
#        [0, 1, 0, 1, 0],
#        [0, 0, 1, 0, 0]]

# img = np.ones([5,5])*2
dim = 3

img = [[10, 6, 10, 3, 4],
       [10, 6, 5, 8, 9],
       [10, 6, 10, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]]


pattern = [[0, 1, 0],
           [0, 1, 0],
           [0, 1, 0]]

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
            _s_I += block[i][j]
            _s_I_2 += block[i][j]**2
            _s_P += pattern[i][j]
            _s_P_2 += pattern[i][j]**2
            _s_IP += block[i][j] * pattern[i][j]

    N = n*m
    dem = (N*_s_I_2 - _s_I**2)*(N*_s_P_2 - _s_P**2)
    return abs((N*_s_IP - _s_I*_s_P)/cm.sqrt(dem)) if dem != 0 else 1

# 将第1层的矩阵模版，转换为十进制数字的字符串
def mat_1_to_str(mat):
    (n, m) = _size(mat)

    _b_str = ''
    for i in range(n):
        for j in range(m):
            _b_str += str(int(bin(mat[i][j]), 2))

    _str =str(int(_b_str,2))
    return _str

# tes

# 将字符串解码为矩阵
def str_to_mat(_str,_k):
    # _b_str = bin(int(_str))[2:]
    _len = len(_str)
    _block_size = _len//9
    _mats = []
    if _k == 1:
        _mats = [[_str[dim * i + j] for j in range(dim)] for i in range(dim)]
    else:
        _block_n = dim*dim
        _str_s = [ _str[ (k*_block_n) : (k*_block_n + _block_size)] for k in range(_block_n)]
        _mats = [[ str_to_mat(_str_s[dim*i + j], _k -1) for j in range(dim)] for i in range(dim)]
    return _mats


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





blocks = split_block(img, 3)
p = zncc(blocks[0][0],pattern)

# patterns = generate_patterns()
# p2 = ''
# for i in range(9):
#     p2 += patterns[i]
# # for i in range(9):
# #     p2 += p2
# str1 = str_to_mat(p2,2)
# print(str1[0][1])

# a = mat_1_to_str(pattern)
# a_str = bin(int(a))[2:]
# a_str = '0' + a_str
# b = str_to_mat(a_str,1)
# c = ord(a[1])




