from model_MF import *
from model_NCF import *
from model_GCMC import *
from model_NGCF import *
from model_SCF import *
from model_CGMC import *
from model_LightGCN import *
from model_LCFN import *
from model_LightLCFN import *
from test_model import *
from read_data import *
from print_save import *
import gc

def train_model(para, path_excel, sparse_propagation_matrix):
    [_, _, MODEL, LR, LAMDA, KEEP_PORB, LAYER, EMB_DIM, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, SAMPLE_RATE,
     BATCH_SIZE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_PRETRAIN, IF_TRASFORMATION,
     ACTIVATION, POOLING, N_EPOCH, _, TOP_K] = para
    ## Paths of data
    train_path = DIR + 'train_data.json'
    hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                 # hypergraph embeddings
    graph_embeddings_1d_path = DIR + 'graph_embeddings_1d.json'                     # 1d graph embeddings
    graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                     # 2d graph embeddings
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'     # pretrained latent factors

    ## Load data
    # load training data
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    # load pre-trained embeddings for all deep models
    try:
        pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
    except:
        print('There is no pre-trained feature found!!')
        pre_train_feature = [0, 0]
        IF_PRETRAIN = 0

    # load pre-trained transform bases for LCFN and SGNN
    if MODEL == 'LCFN': hypergraph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    if MODEL == 'LightLCFN':
        if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY)
        if GRAPH_CONV == '2D_graph': graph_embeddings = read_bases(graph_embeddings_2d_path, FREQUENCY_USER, FREQUENCY_ITEM)
        if GRAPH_CONV == '2D_hyper_graph': graph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    ## Define the model
    if MODEL == 'MF':
        model = model_MF(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA)
    if MODEL == 'NCF':
        model = model_NCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                          pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'GCMC':
        model = model_GCMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN,
                           sparse_graph=sparse_propagation_matrix)
    if MODEL == 'NGCF':
        model = model_NGCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN,
                           sparse_graph=sparse_propagation_matrix)
    if MODEL == 'SCF':
        model = model_SCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                          pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN,
                          sparse_graph=sparse_propagation_matrix)
    if MODEL == 'CGMC':
        model = model_CGMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN,
                           sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LightGCN':
        model = model_LightGCN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                               pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN,
                               sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LCFN':
        model = model_LCFN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA,
                           pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN,
                           graph_embeddings=hypergraph_embeddings)
    if MODEL == 'LightLCFN':
        model = model_LightLCFN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER,
                                pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings,
                                graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION,
                                generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN,
                                if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## Split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))
    ## Training iteratively
    F1_max = 0
    F1_df = pd.DataFrame(columns=TOP_K)
    NDCG_df = pd.DataFrame(columns=TOP_K)
    for epoch in range(N_EPOCH):
        for batch_num in range(len(batches) - 1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num + 1]):
                (user, pos_item) = train_data_interaction[sample]
                sample_num = 0
                while sample_num < SAMPLE_RATE:
                    neg_item = int(random.uniform(0, item_num))
                    if not (neg_item in train_data[user]):
                        sample_num += 1
                        train_batch_data.append([user, pos_item, neg_item])
            train_batch_data = np.array(train_batch_data)
            _, loss = sess.run([model.updates, model.loss],
                               feed_dict={model.users: train_batch_data[:, 0],
                                          model.pos_items: train_batch_data[:, 1],
                                          model.neg_items: train_batch_data[:, 2],
                                          model.keep_prob: KEEP_PORB})

        # test the model each epoch
        F1, NDCG = test_model(sess, model)
        F1_max = max(F1_max, F1[0])
        # print performance
        print_value([epoch + 1, loss, F1_max, F1, NDCG])
        # save performance
        F1_df.loc[epoch + 1] = F1
        NDCG_df.loc[epoch + 1] = NDCG
        save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
        if not loss < 10 ** 10:
            break

    del model, loss, _, sess
    gc.collect()
