from __future__ import print_function
from openpyxl import load_workbook

import tensorflow as tf
import statistics

def turkish_to_english_char(word):
    for i in range(len(word)):
        if word[i] == "ç":
            word = word[:i] + "c" + word[i + 1:]
        if word[i] == "ğ":
            word = word[:i] + "g" + word[i + 1:]
        if word[i] == "ı":
            word = word[:i] + "i" + word[i + 1:]
        if word[i] == "ö":
            word = word[:i] + "o" + word[i + 1:]
        if word[i] == "ş":
            word = word[:i] + "s" + word[i + 1:]
        if word[i] == "ü":
            word = word[:i] + "u" + word[i+1:]

    return word

def shorten_word(word):
	word = word[:3]
	return word

def get_normalization(data):
    num_feature = len(data[0])
    col_data_list = list()
    for i in range(num_feature):
        col_data_list.append([])
    for r in range(len(data)):
        for i in range(num_feature):
            col_data_list[i].append(data[r][i])
        #print(r)
    mean_list = list()
    std_list = list()
    for i in range(num_feature):
        mean_list.append(statistics.mean(col_data_list[i]))
        std_list.append(statistics.stdev(col_data_list[i]))
    no_var_list = list()
    for i in range(num_feature):
        if std_list[i] == 0:
            no_var_list.append(i)
            print(i)
    ret_d = dict()
    ret_d["means"] = mean_list
    ret_d["stdevs"] = std_list
    return no_var_list, ret_d

def normalize_data(data, n_d):
    num_rows = len(data)
    num_cols = len(data[0])
    for r in range(num_rows):
        for c in range(num_cols):
            data[r][c] = (data[r][c] - n_d["means"][c]) / n_d["stdevs"][c]
    return data


#get train data and test data from the xlsx
wb = load_workbook(filename = 'train-test-21-features.xlsx')
train_bank = wb['Bank_Train']
test_bank  = wb['Bank_Test']

train_data = list()
test_data = list()

for row in train_bank.rows:
    train_data.append(list())
    for cell in row:
        train_data[-1].append(cell.value)

for row in test_bank.rows:
    test_data.append(list())
    for cell in row:
        test_data[-1].append(cell.value)

train_data = train_data[1:]
test_data = test_data[1:]

#get the scores in another file
scores_train = list()
for i in range(len(train_data)):
    scores_train.append(train_data[i][-1])
    train_data[i] = train_data[i][:-1]

#get Bayesian data

word_dict = dict()

# train data

train_file = open("train_tweets.txt", "r")

lines = train_file.readlines()
train_file.close()

#train "bayesian"

#preprocess and calculate
for line in lines:
    if line[-3] == ".":
        score = float(line[-5:])
    else:
        score = float(line[-3:])
    words = line.rsplit()
    for word in words:
        word = word[:3]
        word = word.lower()
        word = turkish_to_english_char(word)
        if word in word_dict:
            word_dict[word][0] += score
            word_dict[word][1] += 1
        else:
            word_dict[word] = list()
            word_dict[word].append(score)
            word_dict[word].append(1)

#get scores of each word
for word in word_dict:
    total_score = word_dict[word][0]
    total_instance = word_dict[word][1]
    word_dict[word] = total_score / (total_instance+1)

#add "bayesian" scores to train_data

for i in range(len(lines)):
    line = lines[i]
    words = line.rsplit()
    #get rid of score
    words = words[:-1]
    total_w = len(words)
    total_pos = 0
    for word in words:
        word = turkish_to_english_char(word)
        word = shorten_word(word)
        if word in word_dict:
            total_pos += word_dict[word]
    score_calculated = total_pos / total_w
    train_data[i].append(score_calculated)

'''-------------------------------------------------'''

#open test file and get the bayesian input
file_test = open("test_tweets.txt", "r")

lines = file_test.readlines()
file_test.close()

scores_test = list()

for i in range(len(lines)):
    line = lines[i]
    words = line.rsplit()
    score = words[-1]
    scores_test.append(float(score))
    words = words[:-1]
    total_w = len(words)
    total_pos = 0
    for word in words:
      word = turkish_to_english_char(word)
      word = shorten_word(word)
      if word in word_dict:
        total_pos += word_dict[word]
    score_calculated = total_pos / total_w
    test_data[i].append(score_calculated)

normalization = dict()
null_l, normalization = get_normalization(train_data)

for i in range(len(null_l)):
    del normalization["means"][null_l[i] - i]
    del normalization["stdevs"][null_l[i] - i]

for r in range(len(train_data)):
    for i in range(len(null_l)):
        train_data[r] = train_data[r][:null_l[i]-i] + train_data[r][null_l[i]-i+1:]

for r in range(len(test_data)):
    for i in range(len(null_l)):
        test_data[r] = test_data[r][:null_l[i]-i] + test_data[r][null_l[i]-i+1:]

normalize_data(train_data, normalization)
normalize_data(test_data, normalization)

num_features = len(train_data[0])



# Neural Network

# Parameters
learning_rate = 0.1
num_steps = 1000
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 21 # 21 features
num_classes = 1

# Define the neural network
def neural_net(x_dict):
    x = x_dict['images']
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf._sigmoid(logits)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(labels, dtype=tf.string)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                              global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': word_dict}, y=None, batch_size=1,  num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_file}, y=None, batch_size=1, shuffle=True)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])






































print ("-------")
