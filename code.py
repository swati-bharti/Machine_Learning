import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, average_precision_score, recall_score
import csv
import json
import sys

training_file_name = sys.argv[1]
test_file_name = sys.argv[2]

def printPreview(first,second):
    for i in range(len(first)):
        print(first[i],second[i])

#TRAINING SET PREPARATION

trainSet_text_data = []
reg_trainSet_target = []
class_trainSet_target = []

f = open(training_file_name,'r')
reader = csv.reader(f)
dataset = []
for row in reader:
    dataset.append(row)

all_casts_ids= []
movie_revenues = []
movie_ratings = []
for movie in dataset:
    if(movie[1]=="cast"):
        continue
    casts = json.loads(movie[1])
    movie_casts_ids_string = ""
    for cast in casts:
        if(cast==""):
            continue
        movie_casts_ids_string = movie_casts_ids_string + str(cast['id']) + ','
    if(movie_casts_ids_string):
        movie_casts_ids_string = movie_casts_ids_string[:-1] + ""
    all_casts_ids.append(movie_casts_ids_string)
    movie_revenues.append(movie[13])
    movie_ratings.append(movie[18])

x_casts_ids = np.array(all_casts_ids)

all_crew_names= []
for movie in dataset:
    if(not movie[2] or movie[2]=="crew"):
        continue
    crews = json.loads(movie[2])
    movie_crews_name_string = ""
    for crew in crews:
        if(crew==""):
            continue
        movie_crews_name_string = movie_crews_name_string + str(crew['name']) + ','
    if(movie_crews_name_string):
         movie_crews_name_string= movie_crews_name_string[:-1] + ""
    all_crew_names.append(movie_crews_name_string)
    movie_revenues.append(movie[13])
    movie_ratings.append(movie[18])


x_crews = np.array(all_crew_names)

trainSet_bag_of_words = np.concatenate((x_crews,x_casts_ids),axis=0)
reg_trainSet_target = np.array(movie_revenues)
class_trainSet_target = np.array(movie_ratings)

count = CountVectorizer()
trainSet_bag_of_words = count.fit_transform(trainSet_bag_of_words)

X_train = trainSet_bag_of_words
y_train_reg = reg_trainSet_target
y_train_class = class_trainSet_target


#TESTSET PREPARATION

trainSet_text_data = []
reg_trainSet_target = []
class_trainSet_target = []

f = open(test_file_name,'r')
reader = csv.reader(f)
dataset = []
for row in reader:
    dataset.append(row)

all_casts_ids= []
movie_revenues = []
movie_ratings = []
for movie in dataset:
    if(not movie or movie[1]=="cast"):
        continue
    casts = json.loads(movie[1])
    movie_casts_ids_string = ""
    for cast in casts:
        if(cast==""):
            continue
        movie_casts_ids_string = movie_casts_ids_string + str(cast['id']) + ','
    if(movie_casts_ids_string):
        movie_casts_ids_string = movie_casts_ids_string[:-1] + ""
    all_casts_ids.append(movie_casts_ids_string)
    movie_revenues.append(movie[13])
    movie_ratings.append(movie[18])

x_casts_ids = np.array(all_casts_ids)

all_crew_names= []
reg_movie_id = []
for movie in dataset:
    if(not movie or movie[2]=="crew"):
        continue
    crews = json.loads(movie[2])
    movie_crews_name_string = ""
    for crew in crews:
        if(crew==""):
            continue
        movie_crews_name_string = movie_crews_name_string + str(crew['name']) + ','
    if(movie_crews_name_string):
        movie_crews_name_string= movie_crews_name_string[:-1] + ""
    all_crew_names.append(movie_crews_name_string)
    movie_revenues.append(movie[13])
    movie_ratings.append(movie[18])
    reg_movie_id.append(movie[0])

x_crews = np.array(all_crew_names)

testSet_bag_of_words = np.concatenate((x_crews,x_casts_ids),axis=0)

reg_testSet_target = np.array(movie_revenues)
class_trainSet_target= np.array(movie_ratings)

testSet_bag_of_words = count.transform(testSet_bag_of_words)


X_test = testSet_bag_of_words
reg_y_test = reg_testSet_target
class_y_test = class_trainSet_target

print(X_train.shape,y_train_class.shape,X_test.shape,class_y_test.shape)

#Linear regression
reg = LinearRegression()
reg_model = reg.fit(X_train,y_train_reg)
predicted_y = reg_model.predict(X_test)

X = predicted_y 
Y= reg_y_test 

X = list(map(int,X))
Y = list(map(int,Y))


for i in range(len(X)):
    if(X[i]<0):
        X[i] = abs(X[i])

corr2 = np.corrcoef(X,Y)[1][0]
print("np corr for reg : ",corr2)

#MSR


msr = mean_squared_error(X,Y)
print("msr for reg : ",msr)

zid = 'z5277828'
filename = zid+'.PART1.summary.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["zid", "MSR", "correlation"])
    writer.writerow([zid, str(msr), str(corr2)])

zid = 'z5277828'
filename = zid+'.PART1.output.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["movieID", "predicted_revenue"])
    for i in range(len(reg_movie_id)):
        writer.writerow([reg_movie_id[i], str(X[i])])

#classification

trainSet_text_data = []
trainSet_target = []

clf = MultinomialNB()
mnb_model = clf.fit(X_train, y_train_class)
predicted_y = mnb_model.predict(X_test)
print("----===MNB===----")
acc = accuracy_score(class_y_test, predicted_y)
print("Accuracy : ",acc)
report = classification_report(class_y_test, predicted_y,output_dict=True)
avg_precision =  report['weighted avg']['precision'] 
avg_recall = report['weighted avg']['recall']  

print("average recall : ",avg_recall)
print("average precision",avg_precision)


zid = 'z5277828'
filename = zid+'.PART2.summary.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["zid", "average_precision", "average_recall","accuracy"])
    writer.writerow([zid, str(avg_precision), str(avg_recall),str(acc)])

zid = 'z5277828'
filename = zid+'.PART2.output.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["movieID", "predicted_rating"])
    for i in range(len(reg_movie_id)):
        writer.writerow([reg_movie_id[i], str(predicted_y[i])])

