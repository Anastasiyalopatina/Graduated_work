import math
import numpy as np
import pandas as pd
import sys
import random
import time
import statistics
from sklearn.preprocessing import LabelEncoder
import xlsxwriter
from sklearn.cluster import KMeans
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.spatial import distance

class kNN():

    def __init__(self,
                 k = 2,
                 weighted = True,
                 metric='euclidean'):
        super().__init__()

        self.metric = metric
        self.k = int(k)
        self.weighted = bool(weighted)

    def fit(self, X, r, y=None):
        self.distr, self.cent = self.k_means(X, r)
        return self

    def predict(self, X):
        self.scores = self.decision_function(X)
        self.scores_to_labels(X)
        return self.labels

    def fit_predict(self, X, r, y=None):
        self.fit(X, r, y)
        self.predict(X)
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
            return D[:, -1].flatten()  

    def distance(self, x, y):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x-y)**2))  
        
        elif self.metric == 'chebyshev':
            return distance.chebyshev(x, y)

        elif self.metric == 'manhetten':
            return distance.cityblock(x, y)
        
        elif self.metric == 'minkowski':
            return distance.minkowski(x, y, 2)  

    def k_means(self, X, c):

        kmeans = KMeans(n_clusters=c, random_state=0).fit(X)
        centroids = kmeans.cluster_centers_

        distr = np.zeros((len(X), 3))
        distr[:, 0] = X[:, 0]
        distr[:, 1] = kmeans.predict(X)

        for i in range(len(X)):
            distr[i, 2] = self.distance(distr[i, 0], centroids[int(distr[i, 1])]) 

        distr = distr[distr[:, 2].argsort()[::-1]]

        return distr, centroids

    def search_kNN(self, X, distr, cent):

        distn = np.zeros((len(X), self.k))

        for i in range(len(X)):
            c_d = np.zeros((len(cent), 2))
            c_d[:, 0] = np.arange(len(cent))
            knn = []

            for j in range(len(cent)):
                c_d[j, 1] = self.distance(X[i], cent[j])
            c_d = c_d[c_d[:, 1].argsort()]

            for temp_cl in range(len(cent)):
                P = distr[distr[:, 1] == c_d[temp_cl, 0]]
                
                for j in range(len(P)):
                    if len(knn) < self.k:
                        d = self.distance(X[i], P[j, 0])
                        knn.append([P[j, 0], d])

                    else:
                        d_max = max(knn[:][1])
                        if d_max <= abs(c_d[temp_cl, 1] - P[j, 2]):
                            break
                        else:
                            d = self.distance(X[i], P[j, 0])
                            if d_max > d:
                                m = np.where(d_max)[0]
                                knn[m[0]][0] = P[j, 0]
                                knn[m[0]][1] = d

            distn[i, :] = knn[:][1]

        return distn

def generate_anomaly(X, anomaly_rate):
    # -1 - anomaly, 1 - not anomaly
    X_anomaly = X.copy()
    nr, nc = X_anomaly.shape
    list_rand = [-1, 1]
    C = np.ones((nr, nc))
    for i in range(nc):
        C[:, i] = np.array(random.choices(list_rand, weights = [anomaly_rate, 1-anomaly_rate], k = nr))
    
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

df = pd.read_csv('supermarket_sales_k.csv')
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
res = np.zeros(4)

for i in range (0, nc):

    start = time.time()
    detector = kNN(k=2)
    detector.fit(X_gen['X'][0:700, i].reshape(-1, 1), 6)
    knn_sample[:, i] = detector.predict(X[:, i].reshape(-1, 1))
res = results(knn_sample, C)

pd.DataFrame(knn_sample).to_excel(writer, sheet_name='kNN sample')
pd.DataFrame(res).to_excel(writer, sheet_name='Res')
pd.DataFrame(X_truth).to_excel(writer, sheet_name='Initial_data')
pd.DataFrame(X).to_excel(writer, sheet_name='Data_with_anomaly')
pd.DataFrame(C).to_excel(writer, sheet_name='Generate_anomaly')
writer.save()
