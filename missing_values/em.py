import pandas as pd 
import numpy as np
from functools import reduce
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import xlsxwriter
import sys
import random
import math as ma
import scipy.stats as st
import statistics
from sklearn.metrics import mean_squared_error
import copy

writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')

def generate_nan (X, nan_perc):
    X_full = X.copy()
    nr, nc = X_full.shape
    list_rand = [True, False]
    C = np.ones((nr, nc), dtype=bool)
    for i in range(nc):
        if i == 2 or i == 3 or i == 4 or i == 5:
            C[:, i] = np.array(random.choices(list_rand, weights = [100-nan_perc, nan_perc], k = nr))
    check = np.where(sum(C.T) == 0)[0] 
    if len(check) == 0:
        X_full[C == False] = np.nan
    else:
        for i in check:
            repl = np.random.choice(
                nc, 
                int(np.ceil(nc * np.random.random_sample())), 
                replace = False
            )
            C[i, np.ix_(repl)] = True
        X_full[C == False] = np.nan
    
    result = {
        'X': X_full,
        'C': C,
        'nan_perc': nan_perc
    }
    return result

def em(X, max_iter = 3000, eps = 1e-03):    
    nr, nc = X.shape
    C = np.isnan(X) == False
    nc_1 = np.arange(1, nc + 1, step = 1)
    M = nc_1 * (C == False) - 1 
    O = nc_1 * C - 1 
    
    Mu = np.nanmean(X, axis = 0)
    rows_o = np.where(np.isnan(sum(X.T)) == False)[0] 
    S = np.cov(X[rows_o, ].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))
    
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i,]) != set(nc_1 - 1):  
                M_i, O_i = M[i,][M[i,] != -1], O[i,][O[i,] != -1]
                MM = S[np.ix_(M_i, M_i)]
                MO = S[np.ix_(M_i, O_i)]
                OM = MO.T
                OO = S[np.ix_(O_i, O_i)]
                if np.linalg.cond(OO) < 1 / sys.float_info.epsilon:
                    Mu_tilde[i] = Mu[np.ix_(M_i)] + MO @ np.linalg.inv(OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                    X_tilde[i, M_i] = Mu_tilde[i]
                    MM_O = MM - MO @ np.linalg.inv(OO) @ OM
                else:
                    Mu_tilde[i] = Mu[np.ix_(M_i)] + MO @ np.linalg.pinv(OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                    X_tilde[i, M_i] = Mu_tilde[i]
                    MM_O = MM - MO @ np.linalg.pinv(OO) @ OM
                S_tilde[i][np.ix_(M_i, M_i)] = MM_O

        Mu_new = np.mean(X_tilde, axis=0)
        S_new = np.cov(X_tilde.T, bias=1) + reduce(np.add, S_tilde.values()) / nr
        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or np.linalg.norm(S - S_new, ord=2) >= eps
        Mu = Mu_new
        S = S_new
        iteration += 1

    result = {
        'X_imputed': X_tilde,
        'C': C,
    }

    return result

def round_not_num (X_init, X_new, types_bytes):
    nr, nc = X_init.shape
    for j in range (0, nc):
        if types_bytes[j] == 0:
            max_col = max(X_init[:, j])
            min_col = min(X_init[:, j])
            for i in range (0, nr):
                X_new[i, j] = round(X_new[i, j])
                if X_new[i, j] > max_col:
                    X_new[i, j] -= 1
                if X_new[i, j] < min_col:
                    X_new[i, j] += 1
    return X_new

def mcar_test(data):

    dataset = data.copy()
    vars = dataset.dtypes.index.values
    n_var = dataset.shape[1]
    gmean = dataset.mean()
    gcov = dataset.cov()

    r = 1 * dataset.isnull()
    mdp = np.dot(r, list(map(lambda x: ma.pow(2, x), range(n_var))))
    sorted_mdp = sorted(np.unique(mdp))
    n_pat = len(sorted_mdp)
    correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
    dataset['mdp'] = pd.Series(correct_mdp, index=dataset.index)

    pj = 0
    d2 = 0
    for i in range(n_pat):
        dataset_temp = dataset.loc[dataset['mdp'] == i, vars]
        select_vars = ~dataset_temp.isnull().any()
        pj += np.sum(select_vars)
        select_vars = vars[select_vars]
        means = dataset_temp[select_vars].mean() - gmean[select_vars]
        select_cov = gcov.loc[select_vars, select_vars]
        mj = len(dataset_temp)
        parta = np.dot(means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1])))
        d2 += mj * (np.dot(parta, means))

    df = pj - n_var
    p_value = 1 - st.chi2.cdf(d2, df)

    return p_value

df = pd.read_csv('data_p.csv')
ldl_encoder = LabelEncoder()

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
    df.iloc[:, i] = df.iloc[:, i].astype('float64')

X_truth = df.iloc[0:1000, :].to_numpy()
X_gen = generate_nan(X_truth, nan_perc = 10) 
X = X_gen['X'].copy()
C = X_gen['C'].copy()
X_df = pd.DataFrame.from_dict(X)
nr, nc = X.shape

result = em(X)
res_em = round_not_num(X, result['X_imputed'], types_bytes)

mean = np.zeros((2, nc))
var = np.zeros((2, nc))
rmse = ma.sqrt(mean_squared_error(X_truth, res_em))

for i in range (0, nc):
    mean[0, i] = statistics.mean(X_truth[:, i])
    mean[1, i] = statistics.mean(res_em[:, i])

    var[0, i] = statistics.variance(X_truth[:, i])
    var[1, i] = statistics.variance(res_em[:, i])

pd.DataFrame(mean).to_excel(writer, sheet_name='Mean')
pd.DataFrame(var).to_excel(writer, sheet_name='Var')
pd.DataFrame(rmse).to_excel(writer, sheet_name='RMSE')
pd.DataFrame(res_em).to_excel(writer, sheet_name='EM')
pd.DataFrame(df).to_excel(writer, sheet_name='Dataset')
pd.DataFrame(X).to_excel(writer, sheet_name='X')
writer.save()
