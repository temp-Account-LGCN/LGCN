from evaluation import *
from read_data import *
from params import DIR
from params import TOP_K
from params import TEST_VALIDATION
from params import TEST_USER_BATCH
import random as rd
import multiprocessing
import gc
cores = multiprocessing.cpu_count()

train_path = DIR+'train_data.json'
test_path = DIR+'test_data.json'
validation_path = DIR+'validation_data.json'

## load data
[train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
teat_vali_path = validation_path if TEST_VALIDATION == 'Validation' else test_path
test_data = read_data(teat_vali_path)[0]
score_min = -10 ** 5
def test_one_user(user, top_item):
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    top_item = top_item.tolist()  ## make testing fatser
    for i in range(k_num):
        f1[i] = evaluation_F1(top_item, TOP_K[i], test_data[user])
        ndcg[i] = evaluation_NDCG(top_item, TOP_K[i], test_data[user])
    return f1, ndcg

def test_model(sess, model):
    ## Since Amazon is too large to calculate user_num*item_num interactions, we select TEST_USER_BATCH users to test the model.
    ## For some heavy models (e.g., NCF and LightLCFN with MLP as the predictor), calculating TEST_USER_BATCH*item_num interactions is still space-consuming, we split TEST_USER_BATCH users into mini batches further
    user_top_items = np.zeros((TEST_USER_BATCH, max(TOP_K))).astype(dtype=int32)
    test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num))
    mini_batch_list.append(TEST_USER_BATCH)
    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1]
        user_batch = test_batch[u1: u2]
        items_in_train_data = np.zeros((u2 - u1, item_num))
        for u_index, user in enumerate(user_batch):
            for item in train_data[user]:
                items_in_train_data[u_index, item] = score_min

        user_top_items_batch = sess.run(model.top_items, feed_dict={model.users: user_batch,
                                                                    model.keep_prob: 1,
                                                                    model.items_in_train_data: items_in_train_data,
                                                                    model.top_k: max(TOP_K)})
        user_top_items[u1: u2] = user_top_items_batch
    result = []
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            result.append(test_one_user(user, user_top_items[u_index]))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    del result, user_top_items_batch, user_top_items
    gc.collect()
    return F1, NDCG
