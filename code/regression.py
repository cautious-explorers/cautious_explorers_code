import pickle
def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f) 
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import tqdm as tq
import datetime
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as st

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor



#01-99
def get_dummy(datak):
    list100=[]
    aidlist=[]
    for i,g in datak.groupby(['aid']):
        aidlist.append(i)
        
        this_genres = [np.nan]*100
        i=list(g.first_genre)[0]
        i=i[1:-1]
        i=i.replace('\'', '') 
        i=i.replace(' ', '') 
        i=i.split(',')
        for t_i in i:
            this_genres[int(t_i[:2])]=1
        list100.append(this_genres)
    
    aid_genre=pd.DataFrame( list100  )
    aid_genre['aid'] = aidlist
    datak=datak.merge(aid_genre,on=['aid']).drop(columns='first_genre')
    return datak


def cal_norm_k(k, std_x, std_y):
    return k * std_x / std_y

def draw_relation_explore_score_each(attribute_x, attribute_y1, attribute_y2, width, till_rate, data):
    director_count = []
    strictly_switch_prob_mean_1 = []
    strictly_switch_prob_mean_2 = []
    strictly_switch_prob_std = []
    low_list_1 = []
    high_list_1 = []
    low_list_2 = []
    high_list_2 = []

    x_labels = np.arange(data[attribute_x].min(), data[attribute_x].max(), width)

    for lower_rating in x_labels:

        higher_rating = lower_rating + width

        required_dir = data.loc[(data[attribute_x] >= lower_rating) & (data[attribute_x] < higher_rating)]
        director_count.append(len(required_dir))
        strictly_switch_prob_mean_1.append(required_dir[attribute_y1].mean())
    
        strictly_switch_prob_mean_2.append(required_dir[attribute_y2].mean())

    fig=plt.figure(figsize=(5, 3))
    ax1=fig.add_subplot(111)

    till=int(len(x_labels)*till_rate)
    line1 = ax1.scatter(x_labels[:till], strictly_switch_prob_mean_1[:till], label=attribute_y1, color = 'r')
    line2 = ax1.plot(x_labels[:till], strictly_switch_prob_mean_2[:till], label=attribute_y2, color = 'g')
    ax1.legend(loc='best')
    plt.xlabel(attribute_x)
    ax1.set_ylabel(attribute_y1)

    ax2=ax1.twinx()
    ax2.bar(x_labels[:till], director_count[:till], label='Count', alpha=0.2, width = width)
    ax2.set_ylabel('Count')
    plt.show()
    

def mape(y_true, y_pred): 
    return np.mean(np.abs((y_pred - y_true) / [max(i,0.00000001) for i in y_true])) 
def cal_norm_k(k, std_x, std_y):
    return k * std_x / std_y


def draw_distribution(attribute1, attribute2, width, till_rate, data):
    fig=plt.figure(figsize=(5, 3))
    ax1=fig.add_subplot(111)
    

    for attr in [attribute1, attribute2]:
        director_count = []
        x_labels = np.arange(data[attr].min(), data[attr].max(), width)

        for lower_rating in x_labels:

            higher_rating = lower_rating + width
            required_dir = data[(data[attr] >= lower_rating) & (data[attr] < higher_rating)]
            director_count.append(len(required_dir))

        till=int(len(x_labels)*till_rate)
        ax1.bar(x_labels[:till], director_count[:till], label=attr, alpha=0.2, width = width)
    
    ax1.set_ylabel('Count')
    plt.xlabel(attribute1)
    plt.legend()
    plt.show()
    
    
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy import stats


def cal_norm_k(k, std_x, std_y):
    return k * std_x / std_y


def reg_and_pre(poy, file_dir, select, attribute_y, attributes, dummy_attris, summary,range_list):
    N = []
    r2 = []
    mse = []
    rmse = []
    mae = []

    mu_bar = []
    mu = []
    std_bar = []
    std = []
    
    pvalue = {}
    norm_k_order = []
    coeff = {}
    norm_coeff = {}
    err = {}
    norm_err = {}

    if poy == 'p':
        f_str = "N_{}_author_info.csv"
        
    elif poy == 'cy':
        f_str = "cyear_{}_author_info.csv"

    for cut_year in range_list:
        
        predict=pd.read_csv(file_dir+f_str.format(cut_year))
        
        if select:
            if poy == 'y':
                predict = predict[(predict.past_paperCount>=5)&(predict.post_paperCount>=3)]
                behave_list = ['past_logCit','past_paperCount']
            if poy == 'p':
                predict = predict[(predict.post_paperCount>=3)]
                behave_list = ['past_logCit']
            if poy == 'cy':
                predict = predict[(predict.past_paperCount>=5)&(predict.post_paperCount>=3)]
                behave_list = ['past_logCit','past_paperCount']
        
        # print(len(predict))

        if attributes:
            x_label_1 = attributes + behave_list
            
        else:
            x_label_1 = behave_list
            
        predict = predict[x_label_1+['aid','first_genre','first_year']+[attribute_y]]  # 这两个是之后要删掉的，所以放在这里  
        predict = predict.dropna()
        if len(predict)==0:
            continue
            
        predict = get_dummy(predict)
        predict = pd.concat((predict, pd.get_dummies(predict['first_year'], drop_first=True)), axis=1).drop(columns='first_year')
        if attributes:
            attributes_used = attributes.copy()
        if dummy_attris:
            for attri in dummy_attris:
                dummy_df = pd.get_dummies(predict[attri]).drop(columns=max(predict[attri]))
                predict = pd.concat((predict, dummy_df), axis=1).drop(columns=attri)
                attributes_used += list(dummy_df.columns)
                attributes_used.remove(attri)
        
        # print(attributes)
        predict=predict.dropna(axis=1,how='all')
        predict.fillna(0,inplace=True)
        # print(predict.describe())
        
        x_label = list(set(predict.columns) - set(['aid',attribute_y]))
        # print(x_label)
        X_train=predict[x_label]
        X_train=sm.add_constant(X_train)
        Y_train=predict[attribute_y]
        # print(X_train[['A','B','C']].head())
        # print(np.asarray(X_train))
        est = sm.OLS(Y_train , X_train).fit()
        
        
        if summary:
            print(est.summary())
        
        if attributes:
            for a in attributes_used:
                if a not in coeff.keys():
                    pvalue[a] = [float(est.pvalues[a])]
                    coeff[a] = [float(est.params[a])]
                    norm_coeff[a] = [cal_norm_k(float(est.params[a]), X_train[a].std(), Y_train.std())]
                    err[a] = [float(est.bse[a])]
                    norm_err[a] = [cal_norm_k(float(est.bse[a]), X_train[a].std(), Y_train.std())]
                else:
                    pvalue[a].append(float(est.pvalues[a]))
                    coeff[a].append(float(est.params[a]))
                    norm_coeff[a].append(cal_norm_k(float(est.params[a]), X_train[a].std(), Y_train.std()))
                    err[a].append(float(est.bse[a]))
                    norm_err[a].append(cal_norm_k(float(est.bse[a]), X_train[a].std(), Y_train.std()))
        N.append(est.nobs)    
        r2.append(est.rsquared_adj)
    
    parameters = {'r2':r2, 'N':N, 'mse':mse, 'rmse':rmse, 'mae':mae, 'mu_bar':mu_bar, 'mu':mu, 'std_bar':std_bar, 'std':std, 'pvalue':pvalue, 'norm_k_order':norm_k_order,'coeff':coeff,'norm_coeff':norm_coeff, 'err':err, 'norm_err':norm_err}
    print(parameters.keys())
    
    return parameters, est


def test(para):
    r2, N, mse, rmse, mae, mu_bar, mu, std_bar, std, pvalue, norm_k_order,coeff,norm_coeff,err, norm_err = para[0].values()
    for key in coeff.keys():
        print('%s回归系数为正'%key,'%.2f'%(sum([c>0 for c in coeff[key]])/len(coeff[key])))
        print('%s回归系数显著<=0.05'%key,'%.2f'%(sum([p<0.05 for p in pvalue[key]])/len(pvalue[key])))
    print('预测R2平均%.2f范围%.2f-%.2f'%(np.mean(r2),np.min(r2),np.max(r2)))