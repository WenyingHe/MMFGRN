#!|usr|bin|env python
# encoding:utf-8
# @time: 12:19
# @author: hwy
# @file: Fusion_model.py
# @software: PyCharm
import csv
import random
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from M3_xgb import get_final_score_ODE



def single_col_timeseries(scol, split, timelag, maxlag):
    slen = int(len(scol) / split)
    res_col = pd.Series()

    for index in range(0, split):
        tmp = scol[int(index * slen + timelag):(slen + index * slen + timelag - maxlag)]
        res_col = res_col.append(tmp, ignore_index=True)

    return res_col



def invert_expression_timeseries(exp_mat, split, maxlag, pan=0):
  
    df = pd.DataFrame()
    all_mean = np.mean(exp_mat.values)
    all_std = np.std(exp_mat.values)

    for index in range(0, len(exp_mat.columns)):
        sname = exp_mat.columns[index];
        # df[sname] = exp_mat[sname]
        for jindex in range(0 + pan, maxlag + pan):
            # use random to change the sequence
            df[(sname + '_' + str(maxlag - jindex))] = single_col_timeseries(exp_mat[sname], split, jindex, maxlag)

    return df



def get_links(target_name, name, importance_, inverse=False):
   
    feature_imp = pd.DataFrame(importance_, index=name, columns=['imp'])

    #
    feature_large_set = {}  # 

    for i in range(0, len(feature_imp.index)):
        tmp_name = feature_imp.index[i].split('_')
        if tmp_name[0] != target_name:
            if not inverse:
                if (tmp_name[0] + "\t" + target_name) not in feature_large_set:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[tmp_name[0] + "\t" + target_name] = tf_score
                else:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[tmp_name[0] + "\t" + target_name] = max(
                        feature_large_set[tmp_name[0] + "\t" + target_name], tf_score)
            else:
                if (target_name + "\t" + tmp_name[0]) not in feature_large_set:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[target_name + "\t" + tmp_name[0]] = tf_score
                else:
                    tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
                    feature_large_set[target_name + "\t" + tmp_name[0]] = max(
                        feature_large_set[target_name + "\t" + tmp_name[0]], tf_score)

    return feature_large_set



def compute_score1_dict1(data, samples, timelag):
    score_1 = []
    dict_1 = []
    for index in range(0, len(data.columns)):

        t_data = data.copy()
        y = single_col_timeseries(data[data.columns[index]], split=samples, timelag=timelag, maxlag=timelag)
        y_normal = (y - np.mean(y)) / np.std(y)
        x_c = invert_expression_timeseries(t_data, split=samples, maxlag=timelag, pan=0)
        clfx = LGBMRegressor(boosting_type='dart',
                             objective='regression',
                             max_depth=2,
                             num_leaves=50,
                             min_child_samples=17,
                             min_child_weight=0.001,
                             learning_rate=0.009,
                             subsample=0.8,
                             n_estimators=1000,
                             colsample_bytree=0.6,
                             # reg_alpha=0,
                             importance_type='gain')
        clfx.fit(x_c, y_normal)
        err_1 = mean_squared_error(clfx.predict(x_c), y_normal)
        _importance_per = clfx.feature_importances_
        tmp_large = get_links(data.columns[index], x_c.columns.values, _importance_per)
        score_1.append(err_1)
        dict_1.append(tmp_large)

    return dict_1, score_1



def compute_score2_dict2(data, samples, timelag):

    score_2 = []
    dict_2 = []
    for index in range(0, len(data.columns)):
        t_data = data.copy()
        y = single_col_timeseries(data[data.columns[index]], split=samples, timelag=0, maxlag=timelag)
        y_normal = (y - np.mean(y)) / np.std(y)
        x_c = invert_expression_timeseries(t_data, split=samples, maxlag=timelag, pan=1)
        clf = LGBMRegressor(boosting_type='dart',
                             objective='regression',
                             max_depth=2,
                             num_leaves=50,
                             min_child_samples=17,
                             min_child_weight=0.001,
                             learning_rate=0.009,
                             subsample=0.8,
                             n_estimators=1000,
                             colsample_bytree=0.6,
                             # reg_alpha=0,
                             importance_type='gain')
        clf.fit(x_c, y_normal)
        err_2 = mean_squared_error(clf.predict(x_c), y_normal)
        _importance_per_re = clf.feature_importances_
        tmp_small = get_links(data.columns[index], x_c.columns.values, _importance_per_re, inverse=True)
        score_2.append(err_2)
        dict_2.append(tmp_small)

    return dict_2, score_2



def compute_score3_dict3(SS_data_1_path, SS_data_2_path, SS_data_3_path):
    SS_data_1 = pd.read_csv(SS_data_1_path, sep='\t')  # (10, 10) pd
    SS_data_2 = pd.read_csv(SS_data_2_path, sep='\t')  # (10, 10) pd
    SS_data_3 = pd.read_csv(SS_data_3_path, sep='\t')  # (10, 10) pd
    SS_data = pd.concat([SS_data_1, SS_data_2], axis=0)  # 按行拼接12, (20, 10) pd

    score_3 = []
    dict_3 = []
    for i in range(0, len(SS_data.columns)):
        copy_data = SS_data.copy()  
        y_normal1 = copy_data[copy_data.columns[i]]  
        y_normal = (y_normal1 - np.mean(y_normal1)) / np.std(y_normal1)
        x_c = SS_data.drop(SS_data.columns[i], axis=1)  
        clfx = LGBMRegressor(boosting_type='goss',
                             objective='regression',
                             max_depth=7,
                             num_leaves=30,
                             min_child_samples=1,  
                             min_child_weight=0.001,
                             learning_rate=0.01,
                             subsample=0.5,
                             n_estimators=500,
                             colsample_bytree=0.6,
                             # reg_alpha=0,
                             importance_type='gain')
        clfx.fit(x_c, y_normal)
        err_1 = mean_squared_error(clfx.predict(x_c), y_normal)
        _importance_per = clfx.feature_importances_
        tmp_large = get_links(SS_data.columns[i], x_c.columns.values, _importance_per)
        score_3.append(err_1)
        dict_3.append(tmp_large)

    return dict_3, score_3


def min_max_scalar(score, column):

    index = score.index.values
    score = MinMaxScaler().fit_transform(score)
    score = pd.DataFrame(score, columns=column, index=index)

    return score


def compute_feature_importances(score_1, score_2, score_3, score_4, dicts_1, dicts_2, dicts_3):
 
    dict_all_1 = {}
    dict_all_2 = {}
    dict_all_3 = {}
    score_1 = 1 - score_1 / sum(score_1)
    score_2 = 1 - score_2 / sum(score_2)
    score_3 = 1 - score_3 / sum(score_3)

    length = len(score_1)  
    for score, dicts, dcit_all in zip((score_1, score_2, score_3),
                                      (dicts_1, dicts_2, dicts_3),
                                      (dict_all_1, dict_all_2, dict_all_3)):
        for i in range(length):
            tmp_dict = dicts[i]
            for key in tmp_dict:
                tmp_dict[key] = tmp_dict[key] * score[i]
            dcit_all.update(tmp_dict)

    d1 = pd.DataFrame.from_dict(dict_all_1, orient='index')
    d1.columns = ["score_1"]
    d2 = pd.DataFrame.from_dict(dict_all_2, orient='index')
    d2.columns = ["score_2"]
    d3 = pd.DataFrame.from_dict(dict_all_3, orient='index')
    d3.columns = ["score_3"]

    d1 = min_max_scalar(score=d1, column=['score_1'])
    d2 = min_max_scalar(score=d2, column=['score_2'])
    d3 = min_max_scalar(score=d3, column=['score_3'])
    d4 = score_4['score_4'].tolist()

    all_df_temp = d1.join(d2)
    all_df = all_df_temp.join(d3)
    all_df['score_4'] = d4

    all_df['total'] = np.sqrt(all_df["score_1"] * all_df["score_2"])
    all_df['total_124'] = np.sqrt(all_df["score_1"] * all_df["score_2"] * all_df["score_4"])
    all_df['total_1234'] = np.sqrt(all_df["score_1"] * all_df["score_2"] * all_df["score_3"] * all_df["score_4"])
    all_df['single_steady'] = np.sqrt(all_df["score_3"])
    # all_df['ave'] = (all_df["score_1"] + all_df["score_2"] + all_df["score_3"]+ all_df["score_4"]) / 4

    return all_df


def borda_vote(all_df):

    borda_df = all_df.copy()
    all_df_new = pd.DataFrame(index=borda_df.index.values)
    for temp in ['score_1', 'score_2', 'score_3', 'score_4']:
        borda_df_value = borda_df[temp].tolist()
        sort_1 = sorted(enumerate(borda_df_value), key=lambda x: x[1])  # 元组
        sort_2 = sorted(enumerate(sort_1), key=lambda x: x[1][0])  # 二维元组
        borda_df_value_new = [i for i, v in sort_2]
        all_df_new[temp] = borda_df_value_new
    return all_df_new



def mainRun(expressionFile, knockouts, knockdowns, realdata, samples, outputfile1,timelag=2):

    # data of dict_1/2, score_1/2
    data = pd.read_csv(expressionFile, '\t')
    # data of dict_3, score_3
    SS_data_1_path = knockouts
    SS_data_2_path = knockdowns
    SS_data_3_path = expressionFile
    print('Create input data.')

    dict_1, score_1 = compute_score1_dict1(data=data, samples=samples, timelag=timelag)
    dict_2, score_2 = compute_score2_dict2(data=data, samples=samples, timelag=timelag)
    dict_3, score_3 = compute_score3_dict3(SS_data_1_path, SS_data_2_path, SS_data_3_path)

    path_gene_fake = expressionFile
    path_gene_real = realdata  

    score_4 = get_final_score_ODE(gene_fake=path_gene_fake, gene_real=path_gene_real,
                                  SS_data_1=SS_data_1_path, SS_data_2=SS_data_2_path)
    score_4 = score_4[2]
    score_4 = {'score_4': score_4.values}
    score_4 = pd.DataFrame(score_4)

    all_df = compute_feature_importances(score_1, score_2, score_3, score_4, dict_1, dict_2, dict_3)
    print('Compute feature importance.')
    all_df_new = borda_vote(all_df)
    all_df_new['borda_rank'] = (all_df_new["score_1"] + all_df_new["score_2"] + all_df_new["score_3"]+ all_df_new["score_4"]) / 4
    # 保存基因之间的相关性
    #all_df[['ave']].to_csv(outputfile, sep="\t", header=False, quoting=csv.QUOTE_NONE, escapechar=" ")
    all_df[['total_1234']].to_csv(outputfile1, sep="\t", header=False, quoting=csv.QUOTE_NONE, escapechar=" ")



if __name__ == '__main__':
    path_in_timeseries = "C:/Users/86178/Desktop/GRN_data/DREAM4/insilico_size10/insilico_size10_5_timeseries.tsv"
    path_in_knockouts = "C:/Users/86178/Desktop/GRN_data/DREAM4/insilico_size10/insilico_size10_5_knockouts.tsv"
    path_in_knockdowns = "C:/Users/86178/Desktop/GRN_data/DREAM4/insilico_size10/insilico_size10_5_knockdowns.tsv"
    path_in_real = 'C:/Users/86178/Desktop/GRN_data/DREAM4/insilico_size10/insilico_size10_5_goldstandard.tsv'
    path_out1 = "C:/Users/86178/Desktop/GRN_data/Result/score_FM_5.txt"

    mainRun(expressionFile=path_in_timeseries, knockouts=path_in_knockouts, knockdowns=path_in_knockdowns,
            realdata=path_in_real, samples=5, outputfile1=path_out1)
