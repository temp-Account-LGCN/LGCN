from train_model import *
from params import *
from print_save import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if __name__ == '__main__':
    para = [GPU_INDEX,DATASET,MODEL,LR,LAMDA,EMB_DIM, BATCH_SIZE,SAMPLE_RATE,N_EPOCH,TEST_VALIDATION,TOP_K]
    para_name = ['GPU_INDEX','DATASET','MODEL','LR','LAMDA','EMB_DIM','BATCH_SIZE','SAMPLE_RATE','N_EPOCH','TEST_VALIDATION','TOP_K']
    ## print model hyperparameters
    print_params(para_name, para)
    ## train the model
    train_model(para)

