from train_model import *
from params import *
from print_save import *
import time
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

## constructing sparse propagation matrix
def dense_to_sparse(dense):
    ## converting dense matrices into sparse ones
    idx = np.where(dense != 0)
    val = tf.constant(dense[idx], dtype=np.float32)
    idx = tf.constant(list(map(list, zip(*idx))), dtype=np.int64)
    shp = tf.constant([dense.shape[0],dense.shape[1]], dtype=np.int64)
    sparse = tf.SparseTensor(indices=idx, values=val, dense_shape=shp)
    return sparse

def propagation_matrix_left_norm(graph):
    A = np.zeros([user_num+item_num, user_num+item_num], dtype=np.float32)
    for (user, item) in graph:
        A[user, item + user_num] = 1
        A[item + user_num, user] = 1
    degree = np.sum(A, axis=1, keepdims=False)
    for i in range(len(degree)):
        degree[i] = max(degree[i], 0.1**10)
    A_hat = np.dot(np.diag(np.power(degree, -1)), A)
    spare_A = dense_to_sparse(A_hat)
    return spare_A

def propagation_matrix_sym_norm(graph):
    A = np.zeros([user_num+item_num, user_num+item_num], dtype=np.float32)
    for (user, item) in graph:
        A[user, item + user_num] = 1
        A[item + user_num, user] = 1
    degree = np.sum(A, axis=1, keepdims=False)
    for i in range(len(degree)):
        degree[i] = max(degree[i], 0.1**10)
    temp = np.dot(np.diag(np.power(degree, -0.5)), A)
    A_hat = np.dot(temp, np.diag(np.power(degree, -0.5)))
    spare_A = dense_to_sparse(A_hat)
    return spare_A

if MODEL in ['GCMC', 'SCF', 'CGMC']: sparse_propagation_matrix = propagation_matrix_left_norm(train_data_interaction)
elif MODEL in ['NGCF', 'LightGCN']: sparse_propagation_matrix = propagation_matrix_sym_norm(train_data_interaction)
else: sparse_propagation_matrix = 0

if __name__ == '__main__':
    path_excel = 'experiment_result/'+DATASET+'_'+MODEL+'_'+str(int(time.time()))+str(int(random.uniform(100,900)))+'.xlsx'
    para = [GPU_INDEX, DATASET, MODEL, LR, LAMDA, KEEP_PORB, LAYER, EMB_DIM, FREQUENCY_USER, FREQUENCY_ITEM,
            FREQUENCY, SAMPLE_RATE, BATCH_SIZE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION,
            OPTIMIZATION, IF_PRETRAIN, IF_TRASFORMATION, ACTIVATION, POOLING, N_EPOCH, TEST_VALIDATION, TOP_K]
    para_name = ['GPU_INDEX', 'DATASET', 'MODEL', 'LR', 'LAMDA', 'KEEP_PORB', 'LAYER', 'EMB_DIM', 'FREQUENCY_USER', 'FREQUENCY_ITEM',
                 'FREQUENCY', 'SAMPLE_RATE', 'BATCH_SIZE', 'GRAPH_CONV', 'PREDICTION', 'LOSS_FUNCTION', 'GENERALIZATION',
                 'OPTIMIZATION', 'IF_PRETRAIN', 'IF_TRASFORMATION', 'ACTIVATION', 'POOLING', 'N_EPOCH', 'TEST_VALIDATION', 'TOP_K']
    ## print and save model hyperparameters
    print_params(para_name, para)
    save_params(para_name, para, path_excel)
    ## train the model
    train_model(para, path_excel, sparse_propagation_matrix)

