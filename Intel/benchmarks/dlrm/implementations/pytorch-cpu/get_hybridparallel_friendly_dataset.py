import os
import numpy as np
import time
import math
from tqdm import tqdm
import argparse


def parse(data_file, counts_file, prefix='train', sparse_dense_boundary=2048):
    tar_fea = 1   # single target
    den_fea = 13  # 13 dense  features
    spa_fea = 26  # 26 sparse features
    tad_fea = tar_fea + den_fea
    tot_fea = tar_fea + den_fea + spa_fea
    bytes_per_feature=4
    bytes_per_sample = bytes_per_feature * tot_fea
    data_file_size = os.path.getsize(data_file)
    num_samples = math.ceil(data_file_size / bytes_per_sample)

    print('data file:', data_file, ' counts_file: ',counts_file)
    dir_name = os.path.dirname(data_file)
    data_prefix = data_file.split('/')[-1].split('.')[0]
    counts=[]
    with np.load(counts_file) as data:
        counts = data["counts"]
    
    dense_index = []
    sparse_index = []
    index = 0
    for count in counts:
        if count >= sparse_dense_boundary:
            sparse_index.append(index)
        else:
            dense_index.append(index)
        index += 1
    print(dense_index, " ", sparse_index)

    file_str_list = data_file.split('.')
    sparse_fd_map = dict()
    for spa_index in sparse_index:
        out_file_name = "{}/test/{}_sparse_embedding_index_{}.bin".format(dir_name, data_prefix, spa_index);
        sparse_fd_map[spa_index] = open(out_file_name,'ab+')
    out_file = '{}/test/{}_data_parallel.bin'.format(dir_name, data_prefix,'ab+')
    out_file_fd = open(out_file, 'ab+')

    with open(data_file, 'rb') as file:
         for idx in tqdm(range(num_samples)):
             raw_data = file.read(bytes_per_sample)   
             array = np.frombuffer(raw_data, dtype=np.int32)
             dp_data = array[:tad_fea] 
             emb_index = array[tad_fea:tot_fea]
             dp_data = np.append(dp_data, emb_index[dense_index])
             out_file_fd.write(dp_data.tobytes())
             for spa_index in sparse_index:
                 sparse_fd_map[spa_index].write(emb_index[spa_index].tobytes()) 

    out_file_fd.close() 
    for spa_index in sparse_index:
        sparse_fd_map[spa_index].close()
   

parse(data_file='dlrm_dataset/dlrm/input/terabyte_processed_train.bin',counts_file='dlrm_dataset/dlrm/input/day_fea_count.npz')
parse(data_file='dlrm_dataset/dlrm/input/terabyte_processed_val.bin',counts_file='dlrm_dataset/dlrm/input/day_fea_count.npz')
parse(data_file='dlrm_dataset/dlrm/input/terabyte_processed_test.bin',counts_file='dlrm_dataset/dlrm/input/day_fea_count.npz')
