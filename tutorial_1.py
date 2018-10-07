#author: Jason Howe

#invite people for the Kaggle party
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  #Never print matching warnings

# Samples' index started from 1
df_train = pd.read_csv('train.csv', index_col = 'Id')

# #feature label (decoration)
# print(df_train.columns)
#
# '''Take a glance of dataset from Response: Saleprice'''
#
# #discriptive statistics summary
# stats_summary = df_train['SalePrice'].describe()
# print(stats_summary)
#
# #histogram
# Histogram_SalePrice = sns.distplot(df_train['SalePrice'], label='Price_histogram', fit=norm)
# Histogram_SalePrice.set_title('Histogram_SalePrice')
#
# #skewness and kurtosis
# print('skewness:{}'.format(df_train['SalePrice'].skew()))
# print('kurtosis:{}'.format(df_train['SalePrice'].kurt()))
#
# #scatter plot of feature totalbsmtsf/saleprice (NUMERICAL)
# featr = 'TotalBsmtSF'
# scter_data = pd.concat([df_train['SalePrice'],df_train[featr]], axis=1)  # concatenation
# scter_TotalBs = scter_data.plot.scatter(x = featr, y = 'SalePrice', ylim = (0, 800000))
# scter_TotalBs.set_title('Scatter_ TotalBsmtSF & SalePrice')
#
# #scatter plot of feature grlivarea/saleprice (NUMERICAL)
# featr = 'GrLivArea'
# scter_data = pd.concat([df_train['SalePrice'],df_train[featr]], axis=1)  # concatenation
# scter_TotalBs = scter_data.plot.scatter(x = featr, y = 'SalePrice', ylim = (0, 800000))
# scter_TotalBs.set_title('Scatter_ GrLivArea & SalePrice')
#
# # box plot of featureoverallqual/saleprice (CATEGORICAL)
# featr = 'YearBuilt'
# box_data = pd.concat([df_train['SalePrice'],df_train[featr]], axis=1)  # concatenation
# plt.figure(figsize=(8, 6))  # put this ahead of the plot sentence to set the size
# Box_YearBuilt = sns.boxplot(x=featr, y="SalePrice", data=box_data)
# Box_YearBuilt.axis(ymin=0, ymax=800000)
# Box_YearBuilt.set_title('Box_ YearBuilt & SalePrice')
#
# # box plot overallqual/saleprice
# featr = 'OverallQual'
# box_data = pd.concat([df_train['SalePrice'], df_train[featr]], axis=1)
# plt.figure(figsize=(8, 6))
# Box_OverallQual = sns.boxplot(x=featr, y="SalePrice", data=box_data)
# Box_OverallQual.axis(ymin=0, ymax=800000)
# Box_OverallQual.set_title('Box_ OverallQual & SalePrice')
#
# ### corraletion matrix
# corrmat = df_train.corr()
# plt.figure(figsize = (12, 9))
# corrmap = sns.heatmap(corrmat, square = True, center = 0, cmap = 'seismic')
# corrmap.set_title('corraletion matrix')
#
# ### saleprice correlation
#
# # 1. find high correlated variables with SalePrice by corr value
# hcorr_idx_1 = corrmat['SalePrice'][abs(corrmat)['SalePrice'] > 0.6]
# print(corrmat[hcorr_idx_1.index].loc[hcorr_idx_1.index])
#
# # 2. find high correlated variables with SalePrice by corr ranking(the highest 10)
# k = 10
# hcorr_idx_2 = abs(corrmat).nlargest(10, 'SalePrice')['SalePrice'].index
# hcorr_data = df_train[hcorr_idx_2].corr()
# sns.set(font_scale = 1.25)
# plt.figure(figsize = (9, 9))
# hcorr_corrmap = sns.heatmap(hcorr_data, square = True, annot = True, fmt='.2f', \
# annot_kws={'size': 10}, yticklabels=hcorr_idx_2.values, xticklabels=hcorr_idx_2.values)
#
# # multi_scatter plot
# sns.set(style="ticks", color_codes=True)
# sns.pairplot(df_train[hcorr_idx_2[:5]], size = 2, plot_kws={'s':15})
#
#
# ### features(or x) correlation
# corrm_varibs = corrmat.drop('SalePrice').drop('SalePrice', axis = 1)
# hcorr_pairs = pd.DataFrame(data = None, columns = ['x1', 'x2', 'corr'])
#
# # corr filter
# thrshd = 0.6  #threshold value
# corrm_varibs_d = corrm_varibs.shape[0]
# for i in range(len(corrm_varibs)-1):
#     # Because corrm is symmetric matrix we start from no-repeating idx
#     ftr_name = corrm_varibs.columns[i]
#     start = corrm_varibs.index[i + 1]
#     # boolean target
#     col = corrm_varibs[ftr_name][start:]
#     # names of features highly related with present ftr
#     names = col[abs(col) > thrshd].index
#     new_pairs = pd.DataFrame({'x1':[ftr_name]*len(names), 'x2':names, 'corr':col[names]})
#     hcorr_pairs = hcorr_pairs.append(new_pairs, ignore_index = True)
#
# hcorr_pairs.sort_values(axis = 0, ascending = True, by = 'corr')
# print(hcorr_pairs)

## missingdata

total_mis = df_train.isnull().sum().sort_values(ascending = False)
percent_mis = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
data_mis = pd.concat([total_mis, percent_mis], axis=1, keys = ['total', 'percent'])
for i in range(len(total_mis)):
    if total_mis[i] == 0:
        break
missing_data = data_mis[:i]

### dealing with missing data
# delet factors: missing_data['total'] > 1
# delet level: df_train_delmis['Electrical'].isnull()
df_train_delmis = df_train.drop(missing_data[missing_data['total'] > 1].index, axis = 1)
df_train_delmis.drop(df_train_delmis.loc[df_train_delmis['Electrical'].isnull()].index, inplace = True)
# # check that there is no more nan
# df_train_delmis.isnull().sum().max()

### Outliers!
# univariate analysis
# (x-mean)/standard
Scaler = StandardScaler().fit(df_train_delmis['SalePrice'][:, np.newaxis])
saleprice_scaled = Scaler.transform(df_train_delmis['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# bivariate analysis (with saleprice)
# saleprice/grlivarea (delet two points by 2-d scatter plot)
data_sct = pd.concat([df_train_delmis['SalePrice'], df_train_delmis['GrLivArea']], axis = 1)
data_sct.plot.scatter(x = 'GrLivArea', y = 'SalePrice', ylim = (0, 800000))
# delet the outliar from df_train_delmis
del_id = df_train_delmis['GrLivArea'].sort_values()[-2:].index
df_train_delmis.drop(del_id, inplace = True)

### getting hard core
# check the 4 assumptions :
# 1.normality
# 2.heteroscedasticity
# 3.linearity and others
# 4.absence of error correlated error

# histogram and normal probability plot
plt.figure()
sns.distplot(df_train['SalePrice'], fit=norm)
plt.figure()
probplt = stats.probplot(df_train['SalePrice'], plot=plt)












plt.show()  #show all figures above