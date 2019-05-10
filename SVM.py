
from sklearn import svm
import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt


class MovieSVM():

    def __init__(self, threshold):
        self.threshold = threshold
        self.accuracy = []

    def fit(self, A, V):
        T = np.copy(A)
        i = 0
        while(i < len(V)):
        #for i in range(len(V)):
            j = 0
            # print(i)
            while(j<len(V[i])):
            #for j in range(len(V[i])):
                if V[i, j] >= self.threshold:
                    V[i, j] = 1
                elif V[i, j] == 0:
                    V[i, j] = -1
                else:
                    V[i, j] = 0
                # print(i)
                j = j + 1
            i = i + 1
        i=0
        while(i<len(A)):
            j=0
            while(j<len(A[i])):
                if A[i, j] >= self.threshold:
                    T[i, j] = 1
                elif A[i, j] == 0:
                    T[i, j] = -1
                else:
                    T[i, j] = 0
                j+=1
            i+=1
        A = np.copy(T)
        total = np.count_nonzero(V != -2)
        iteration = 0

        svms = []
        length = len(T[0])
        for i in range(length):
            svms.append(svm.SVC())

        self.accuracy = []
        self.recall = []
        self.train_accuracy = []
        self.prec = []
        train_correct = 0
        rec = 0
        total_train = 0
        precition = 0
        positives = 0
        iteration += 1
        i=0
        while(i<len(svms)):
            X = np.delete(A, i, axis=1)
            Y = A[:, i]

            try:
                svms[i].fit(X, Y)
            except:
                dummy = 0

            A[:, i] = svms[i].predict(X)
            j=0
            while(j<len(A[:, i])):
                if(A[j,i]==1):
                    positives+=1
                if(T[j, i] != -1 and A[j, i] != T[j, i]):
                    total_train = total_train+1
                    if(T[j,i]==1):
                        positives += 1
                    if(A[j,i]==1):
                        rec += 1
                elif T[j, i] != -1:
                    if(T[j,i]==1):
                        precition += 1
                        positives+=1
                        rec+=1
                    train_correct = train_correct+1
                    total_train = total_train+1
                j+=1
            i+=1
        self.recall.append((precition*1.0)/rec)
        self.prec.append((precition*1.0)/positives)
        self.train_accuracy.append((train_correct * 1.0) / total_train)
        count = 0
        ActTru = 0
        TruTru = 0
        PredTru = 0
        i=0
        while(i<len(svms)):
            X = np.delete(V, i, axis=1)
            Y = V[:, i]
            Yhat = svms[i].predict(X)

            count += np.sum(Yhat == Y)
            j=0
            while(j<len(Y)):
                if(Y[j]==1):
                    ActTru+=1
                    if(Yhat[j]==1):
                        TruTru +=1
                if(Yhat[j]==1):
                    PredTru+=1
                j+=1
            i+=1
        acc_k = (1.0*count) / total
        self.recall.append((TruTru*1.0)/ActTru)
        self.prec.append((TruTru*1.0)/PredTru)

        self.accuracy.append(acc_k)
        return self.accuracy, self.train_accuracy , self.prec, self.recall

if __name__ == "__main__":
    array = []
    array2 = []
    for i in range(5):
        N = 100+i*200
        print(N)
        Data = util.load_data_matrix()
        V = Data[411:, :N]
        movieSVM = MovieSVM(4.0)
        A = Data[:410, :N]
        Test_M_zero_remove = np.count_nonzero(V)
        for i in range(len(A[0]) - 1, 0, -1):
            if np.count_nonzero(A[:, i]) == 0:
                V = np.delete(V, i, axis=1)
                A = np.delete(A, i, axis=1)
        accuracy, train_accuracy , precition , recall = movieSVM.fit(A, V)
        print(N)
        print("recall :", recall[1])
        print("Final Accuracy Values:", accuracy)
        print("Precision :", precition[1])
       ## print("Training Accuracy:", train_accuracy)
        array.append(accuracy)
        f1score = (2*(recall[1])*precition[1])/(precition[1]+recall[1])
        array2.append(f1score)
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.plot(array)
    plt.show()
