import math
import numpy as np
import pandas as pd
import sys
import random
import time
import statistics
from sklearn.preprocessing import LabelEncoder
import xlsxwriter
from scipy import special
from sklearn.cluster import KMeans

class kNN():

    def __init__(self,
                 k = 2,
                 weighted = True,
                 metric='euclidean'):
        super().__init__()

        self.metric = metric
        self.k = int(k)
        self.weighted = bool(weighted)

    def fit_predict(self, X_train, X_test, y=None):
        self.scores = self.decision_function(X_train, X_test)
        self.scores_to_labels(X)
        return self.labels

    def scores_to_labels(self, X):
        self.labels = -1*np.ones(len(self.scores), dtype=int)
        nr, _ = X.shape
        self.probs = np.zeros(nr)
        ecdf = ECDF(self.scores)
        for i in range(nr):
            for j in range(nr):
                if self.scores[i] == ecdf.x[j]:
                    self.probs[i] = ecdf.y[j]
        self.labels[self.probs <= 0.9] = 1 

    def probability(self):
        return self.probs

    def decision_function(self, X):
        D = self.search_kNN(X, self.distr, self.cent)
        scores = self.get_distances_by_method(D)
        return scores

    def get_distances_by_method(self, D):
        if self.weighted:
            return np.mean(D[:, 1:], axis=1)
        else:
            D = np.argsort(D, axis=1)
            return D[:, -1].flatten()   

    def distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))    

    def kNN_brute(self, X_train, X_test):

        distn = np.zeros((len(X_test), self.k))

        for i in range (len(X_test)):
            distn_i = np.zeros(len(X_train))
            for j in range(len(X_train)):
                distn_i[j] = self.distance(X_test[i], X_train[j])
            distn_i = distn_i.argsort()

            for m in range (self.k):
                distn[i, m] = distn_i[m]

        return distn

def generate_anomaly(X, anomaly_rate):
    # -1 - anomaly, 1 - not anomaly
    X_anomaly = X.copy()
    nr, nc = X_anomaly.shape
    list_rand = [-1, 1]
    rand = random.choices(list_rand, weights = [anomaly_rate, 1-anomaly_rate], k = nr * nc)
    C  = np.asarray(rand).reshape(nr, nc)
    
    for i in range (0, nr):
        for j in range (0, nc):
            if C[i, j] == -1:
                mul_div = random.randint(0, 1)
                if mul_div == 0:
                    X_anomaly[i,j]*= random.randint(2, 100)
                else:
                    X_anomaly[i,j]/= random.randint(2, 100)
    result = {
        'X': X_anomaly,
        'C': C
    }
    return result

writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')

def results(X, C):
    decision = np.zeros(4)
    nr, nc =  X.shape
    for i in range(0, nr):
        for j in range (0, nc):
            if X[i, j] == -1 and C[i, j] == -1: # true positive
                decision[0] += 1
            if X[i, j] == 1 and C[i, j] == 1: # true negative
                decision[1] += 1
            if X[i, j] == -1 and C[i, j] == 1: # false positive
                decision[2] += 1
            if X[i, j] == 1 and C[i, j] == -1: # false negative
                decision[3] += 1  

    recall = decision[0] / (decision[0] + decision[3])
    accuracy = (decision[0] + decision[1]) / (decision[0] + decision[1] + decision[2] + decision[3])
    precision = decision[0] / (decision[0] + decision[2])
    f = 2*(precision*recall)/(precision+recall)

    return recall, accuracy, precision, f

df = pd.read_csv('supermarket_sales.csv')
ldl_encoder = LabelEncoder()

# преобразование данных в числовой тип данных
types_csv = df.dtypes.to_numpy()
access_list = ['int_', 'intc', 'intp', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64','float_', 
'float16', 'float32', 'float64', 'complex_', 'complex64', 'complex128', 'longdouble', 'clongdouble', 'uint8', 'uint16', 
'uint32', 'uintc','intc']
types_bytes = np.zeros(len(types_csv))
for i in range(0, len(types_csv)):
    for j in range(0, len(access_list)):
        if types_csv[i] == access_list[j]:
            types_bytes[i] = 1 
    if types_bytes[i] == 0:
        df.iloc[:, i] = ldl_encoder.fit_transform(df.iloc[:, i])

X_truth = df.to_numpy()
X_gen = generate_anomaly(X_truth, anomaly_rate = .1)
X = X_gen['X'][700:1000, :].copy()
C = X_gen['C'][700:1000, :].copy()
nr, nc =  X.shape

knn_sample = np.zeros((nr, nc))
res = np.zeros((1,4))
time_arr = np.zeros((1, 5))

for i in range (0, nc):
    print("i:", i)

    start = time.time()
    detector_1 = kNN(k=2)
    knn_sample[:, i] = detector_1.fit_predict(X_gen['X'][0:700, i].reshape(-1, 1), X[:, i].reshape(-1, 1)) 
    finish = time.time()
    time_arr[0,0]+=finish-start
    res[0] = results(knn_sample, C)

pd.DataFrame(knn_sample).to_excel(writer, sheet_name='kNN sample')
pd.DataFrame(time_arr).to_excel(writer, sheet_name='Time')
pd.DataFrame(res).to_excel(writer, sheet_name='Res')
pd.DataFrame(X_truth).to_excel(writer, sheet_name='Initial_data')
pd.DataFrame(X).to_excel(writer, sheet_name='Data_with_anomaly')
pd.DataFrame(C).to_excel(writer, sheet_name='Generate_anomaly')
writer.save()
