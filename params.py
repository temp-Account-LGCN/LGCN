model = 8           # 0:MF, 1:NCF, 2:GCMC, 3:NGCF, 4:SCF, 5:CGMC, 6:Light-GCN, 7:LCFN, 8:LightLCFN
dataset = 0         # 0:Amazon, 1:Movielens
test_validation = 0 # 0:Validate, 1: Test
pred_dim = 128      # predictive embedding dimensionality

## parameters about experiment setting
GPU_INDEX = "0"
DATASET = ['Amazon', 'Movielens'][dataset]
MODEL = ['MF', 'NCF', 'GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN', 'LCFN', 'LightLCFN'][model]

## hyperparameters
LR = [[0.05,0.0002,0.001,0.0001,0.0001,0.0001,0.005,0.0005,0.0005],
      [0.02,0.00001,0.0002,0.00005,0.0001,0.00002,0.0005,0.0005,0.0005]][dataset][model]
LAMDA = [[0.02,0,0.05,0.001,0.02,0.0002,0.02,0.005,0.02],
         [0.01,0,0.02,0.02,0.01,0.05,0.02,0.01,0.1]][dataset][model]
LAYER = [[0,4,1,1,1,1,2,1,1], [0,4,1,1,1,1,2,1,1]][dataset][model]
KEEP_PORB = 0.9
# dimensionality of the embedding layer
EMB_DIM = [pred_dim,int(pred_dim/2),int(pred_dim/(LAYER+1)),int(pred_dim/(LAYER+1)),int(pred_dim/(LAYER+1)),
           int(pred_dim/(LAYER+1)),pred_dim,int(pred_dim/(LAYER+1)),pred_dim][model]
FREQUENCY_USER = [[0,0,0,0,0,0,0,100,100], [0,0,0,0,0,0,0,300,300]][dataset][model]
FREQUENCY_ITEM = [[0,0,0,0,0,0,0,50,50], [0,0,0,0,0,0,0,200,200]][dataset][model]
FREQUENCY = [[0,0,0,0,0,0,0,0,128], [0,0,0,0,0,0,0,0,128]][dataset][model]
SAMPLE_RATE = [[1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1]][dataset][model]
BATCH_SIZE = 10000
TEST_USER_BATCH = [4096, 1024][dataset]

## parameters about model setting (selective for model LightLCFN)
GRAPH_CONV = ['1D', '2D_graph', '2D_hyper_graph'][0]
PREDICTION = ['InnerProduct', 'MLP3'][0]
LOSS_FUNCTION = ['BPR', 'CrossEntropy', 'MSE'][0]
GENERALIZATION = ['Regularization', 'DropOut', 'Regularization+DropOut', 'L2Norm'][0]
OPTIMIZATION = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][2]
IF_PRETRAIN = [False, True][1]
IF_TRASFORMATION = [False, True][0]                           # 0 for not having transformation matrix,1 for having
ACTIVATION = ['None', 'Tanh', 'Sigmoid', 'ReLU'][0]          # select the activation function
POOLING = ['Concat', 'Sum', 'Max', 'Product', 'MLP3'][1]    # select the pooling strategy, the layer of mlp is also changable
if POOLING == 'Concat': EMB_DIM = int(pred_dim/(LAYER+1))
## parameters about model setting
N_EPOCH = 200
TEST_VALIDATION = ['Validation', 'Test'][test_validation]                  # 0:Validate, 1: Test
TOP_K = [2, 5, 10, 20, 50, 100]

DIR = 'dataset/'+DATASET+'/'
