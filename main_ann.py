from openpyxl import load_workbook
import statistics
from sklearn.neural_network import MLPRegressor


def probability2score(prob):
	score = prob * 2 - 1
	return score

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


def probability_score(score):
    return (score + 1) / 2

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
    mean_list = list()
    std_list = list()
    for i in range(num_feature):
        mean_list.append(statistics.mean(col_data_list[i]))
        std_list.append(statistics.stdev(col_data_list[i]))
    no_var_list = list()
    for i in range(num_feature):
        if std_list[i] == 0:
            no_var_list.append(i)
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
        word = turkish_to_english_char(word)
        word = word[:3]
        word = word.lower()
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
bay_train = list()
bay_test = list()

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
    #bay_train.append(score_calculated)


#open test file and get the bayesian input
file_test = open("test_tweets.txt", "r")

lines = file_test.readlines()

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
    #bay_test.append(score_calculated)

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

datafile_train = open("data_train.csv", "w")

for i in range(len(train_data)):
	s = ""
	for val in train_data[i]:
		s += str(val) + ", "
	s += str(scores_train[i])
	s += "\n"
	datafile_train.write(s)

datafile_train.close()

datafile_test = open("data_test.csv", "w")

for i in range(len(test_data)):
	s = ""
	for val in test_data[i]:
		s += str(val) + ","
	s += str(scores_test[i])
	s += " CRLF\n"
	datafile_test.write(s)

datafile_test.close()

ann = MLPRegressor(activation='tanh',solver='sgd',alpha=0.0001,
	hidden_layer_sizes=(100,100,40), random_state=1)

ann.fit(train_data, scores_train)

results = ann.predict(test_data)

error_total = list()

for i in range(len(results)):
	 error_total.append(abs(results[i] - scores_test[i]))

#output scores to resultNN.csv
results_file = open("resultNN.csv", "w")
for i in range(len(scores_test)):
	s = ""
	#words = lines[i].rsplit('	')
	#s += words[0] + '	'
	s += str(results[i])
	s += "\n"
	results_file.write(s)
results_file.close()

print(sum(error_total) / len(error_total))
