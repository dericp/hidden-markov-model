train_path = 'data/twt.train.json'
small_train_path = 'data/twt.train.small.json'
dev_path = 'data/twt.dev.json'
test_path = 'data/twt.test.json'

# constants
UNK = '</UNK>'
START = '</START>'
STOP = '</STOP>'
UNK_THRESHOLD = 0

# HMM params
train_out = 'bi.train.json'
dev_out = 'bi.dev.json'
test_out = 'bi.test.json'
tri_train_out = 'tri.train.json'
tri_dev_out = 'tri.dev.json'
tri_test_out = 'tri.test.json'

LAMBDA_1 = 0.1
LAMBDA_2 = 0.9
UNK_ADD_K = 0.0001

LAMBDA_1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

T_LAMBDA_1 = 0.1
T_LAMBDA_2 = 0.8
T_LAMBDA_3 = 0.1
T_UNK_ADD_K = 1e-5

T_LAMBDA_1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
T_LAMBDA_2s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
