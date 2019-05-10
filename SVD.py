from surprise import accuracy
from surprise import KNNBasic
from surprise import SVD
from surprise import Dataset
import numpy as np
from surprise.model_selection import train_test_split


data = Dataset.load_builtin('ml-1m')


trainset, testset = train_test_split(data, test_size=.25)
length = len(testset)

algo = SVD()

algo.fit(trainset)
predictions = algo.test(testset)


accuracy.rmse(predictions)
accuracy.mae(predictions)
acc = 0
ActualTrue = 0
ActPredTrue = 0
PredTrue = 0
for i in range(length):
    predic = algo.predict(testset[i][0],testset[i][1],testset[i][2])
    if(predic[3]-predic[2]<0.75 and predic[3]-predic[2]>-0.75 ):
        acc+=1
    if(predic[2]>=4):
        ActualTrue +=1
        if(predic[3]>=3.75):
            ActPredTrue +=1
    if(predic[3]>=3.5):
        PredTrue +=1
precition = ((1.0*ActPredTrue)/PredTrue)
recall = ((1.0*ActPredTrue)/ActualTrue)
accuracy = acc/length
print("\nrecall :", recall)
print("\nFinal Accuracy Values:", accuracy)
print("\nPrecision :", precition)