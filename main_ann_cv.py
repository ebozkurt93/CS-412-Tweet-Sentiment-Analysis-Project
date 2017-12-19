from openpyxl import load_workbook
import statistics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
import numpy as np

num_of_words = 0
v_threshold = 0.5
k_fold_num = 10

print("number of words: " + str(num_of_words))
print("variance threshold: " + str(v_threshold))
print(str(k_fold_num) + " folds")


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
	word = word[:5]
	return word


#get train data and test data from the xlsx
wb = load_workbook(filename = 'train-test-21-features.xlsx')
train_bank = wb['Bank_Train']
test_bank  = wb['Bank_Test']

train_data = list()
test_data = list()

#add data to train_data and test_data
for row in train_bank.rows:
    train_data.append(list())
    for cell in row:
        train_data[-1].append(cell.value)
        
for row in test_bank.rows:
    test_data.append(list())
    for cell in row:
        test_data[-1].append(cell.value)

#get the feature names out
train_data = train_data[1:]
test_data = test_data[1:]

#get the scores in another file
scores_train = list()
for i in range(len(train_data)):
    scores_train.append(train_data[i][-1])
    train_data[i] = train_data[i][:-1]

#get the scores from test file

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
    
#treat train_data and test_data as one data
train_data = train_data + test_data
scores_train = scores_train + scores_test

#get average data: This has no risk of overfitting, so using all
  #data would increase our performance. Of course we can try dividing 
  #data but since we would be taking average it would simply be inferior
  #and unnecessary 

word_dict = dict()

train_file = open("train_tweets.txt", "r")
lines = train_file.readlines()
test_file = open("test_tweets.txt", "r")
lines += test_file.readlines()
train_file.close()

#train "bayesian"

#preprocess and calculate
for line in lines:
    words = line.rsplit()
    score = float(words[-1])
    words = words[:-1]
    for word in words:
        word = turkish_to_english_char(word)
        word = shorten_word(word)
        word = word.lower()
        if word in word_dict:
            word_dict[word][0] += score
            word_dict[word][1] += 1
        else:
            word_dict[word] = list()
            word_dict[word].append(score)
            word_dict[word].append(1)
            
word_occurences = list()

for word in word_dict:
	occurrence_word = word_dict[word][1]
	l = [word, occurrence_word, word_dict[word][0]/word_dict[word][1]]
	word_occurences.append(l)

word_occurences = sorted(word_occurences,key=lambda w: w[1], reverse=True)

word_occurences = word_occurences[:num_of_words]

common_word_dict = dict()
for i in range(len(word_occurences)):
	common_word_dict[word_occurences[i][0]] = i

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
    #a list to store if common word is in the tweet
    word_list = list()
    for w in range(num_of_words):
        word_list.append(0)
    for word in words:
        word = turkish_to_english_char(word)
        word = shorten_word(word)
        if word in word_dict:
            total_pos += word_dict[word]
        if word in common_word_dict:
            word_list[common_word_dict[word]] = 1
    for w in range(len(word_list)):
        train_data[i].append(word_list[w])
    score_calculated = total_pos / total_w
    train_data[i].append(score_calculated)
    bay_train.append(score_calculated)

scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)

selector = VarianceThreshold(v_threshold)
train_data = selector.fit_transform(train_data)

datafile = open("data_all.csv", "w")

for i in range(len(train_data)):
	s = ""
	for val in train_data[i]:
		s += str(val) + ", "
	s += str(scores_train[i])
	s += "\n"
	datafile.write(s)

datafile.close()

#Single Neural Network
#ann = MLPRegressor(activation='tanh',solver='sgd',alpha=0.1,
	#hidden_layer_sizes=(30,20), random_state=1, learning_rate_init=0.01)
	
#ann.fit(train_data, scores_train)

#results = ann.predict(test_data)

#NN ensemble

total_data = train_data
total_score = scores_train

kf = KFold(n_splits=k_fold_num)
total_data = np.array(total_data)
total_score = np.array(total_score)

print("Total features: " + str(len(total_data[0])))

nn_list = list()
for i in range(k_fold_num):
	nn_list.append(MLPRegressor(activation='tanh',solver='sgd',alpha=0.1,
		hidden_layer_sizes=(100,20), random_state=1, 
		learning_rate_init=0.01, max_iter=600))


error_list = list()
i = 0
for train_indices, test_indices in kf.split(total_data):
    nn_list[i].fit(total_data[train_indices], total_score[train_indices])
    pred = nn_list[i].predict(total_data[test_indices])
    error = pred - total_score[test_indices]
    error_list.append(sum(np.abs(error))/len(error))
    i+=1

print(sum(error_list) / len(error_list))

pred_list = list()

for i in range(k_fold_num):
	pred_list.append(nn_list[i].predict(total_data))

pred_av = sum(pred_list) / k_fold_num

error_list = np.abs(pred_av - total_score)

print(sum(error_list)/len(error_list))
