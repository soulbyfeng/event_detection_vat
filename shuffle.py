import numpy as np

batch_size = 500
np_index_data = np.random.choice(int(50000 / batch_size), int(50000 / batch_size / 10), replace=False)
print(np_index_data[0:10])
np.save('./random_shuffle.seed',np.array(np_index_data))

np_r=np.load('./random_shuffle.seed.npy')
print(np_r)
for index in np_r:
    print(index)
