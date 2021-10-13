import numpy as np
import numpy.random as npr
from Game import *

score = 0

# a = np.arange(10).reshape(2,5)
# b = np.unravel_index(np.argmax(a, axis=None), a.shape)
# c = np.unravel_index(np.argmin(a, axis=None), a.shape)
#
# d = (b, c)
# print(d)
#
# print(action(d[0], d[1], a))
#


def generate_fix_sum_random_vec(limit, num_elem, tries=10):
    v = np.random.randint(1, 9, num_elem)
    s = sum(v)

    grid_a = np.round(v / s * limit)
    grid_b = np.floor(v / s * limit)
    grid_c = np.ceil(v / s * limit)

    if(np.sum(grid_a) == limit):
        if(np.sum(grid_a == 10) == 0):
            return grid_a
        return generate_fix_sum_random_vec(limit, num_elem, tries-1)
    elif(np.sum(grid_b) == limit):
        if (np.sum(grid_b == 10) == 0):
            return grid_b
        return generate_fix_sum_random_vec(limit, num_elem, tries - 1)
    elif(np.sum(grid_c) == limit):
        if (np.sum(grid_c == 10) == 0):
            return grid_c
        return generate_fix_sum_random_vec(limit, num_elem, tries - 1)
    else:
        return generate_fix_sum_random_vec(limit, num_elem, tries-1)

for i in 10:
    grid = generate_fix_sum_random_vec(900, 170)
    print(grid.reshape(10,17), np.sum(grid))

#
#
#
# test_vec = generate_fix_sum_random_vec(900, 170)
# test_vec = np.array(test_vec.astype(int)).reshape(10, 17)
# dnpTest = (np.array(test_vec)).reshape(10, 17)
# print(dnpTest, "\nsum of vector: ", np.sum(test_vec))
#
#
# print(np.unravel_index(np.argmax(dnpTest, axis=None), dnpTest.shape))

'''
print(a)

print(np.argmax(a, axis=None))

print(np.unravel_index(np.argmax(a, axis=None), a.shape))

print(a[np.unravel_index(np.argmax(a, axis=None), a.shape)])


token = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
word2index = {}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)

new_array = np.zeros((2, 5, 9))  # 10 행 개수, 17 열 개수, 9개 채널 개수(one-hot)
for i, x_value in enumerate(a):
    for j, y_value in enumerate(x_value):
        one_hot_vector = [0] * 9   # 총 9개 채널
        index = word2index[y_value]
        if (index != 0):
            one_hot_vector[index - 1] = 1
        new_array[i, j] = one_hot_vector

print(new_array)
print()
print(np.unravel_index(np.argmax(new_array, axis=None), new_array.shape))
print()
print(new_array[0, 1, 0])
print()
print(new_array[np.unravel_index(np.argmax(new_array, axis=None), new_array.shape)])
'''