#!/usr/bin/env python
# coding: utf-8

# # Capstone Project_ Siddharth Gupta

# ## Data Clean up and descriptive stats

# In[220]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scikitplot as skplot
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier , HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score


# In[221]:


trans_2018 = pd.read_excel('Downloads/Transactions.xls', sheet_name = 'Transactions18')
trans_2019 = pd.read_excel('Downloads/Transactions.xls', sheet_name = 'Transactions19')


# In[222]:


firm_b = pd.read_excel('Downloads/firms_1.xls', sheet_name = 'Remp')


# In[223]:


trans_2019 = trans_2019.rename({'sales_12M' : 'Sales_2019', 'new_Fund_added_12M': 'new_fund_2019'}, axis = 1)


# In[224]:


full_df = pd.merge(trans_2018, trans_2019, on = 'CONTACT_ID')
full_df.head()


# In[225]:


full_df['no_of_sales_12M_1'].fillna(0, inplace = True)


# In[226]:


full_df.loc[:,[i for i in full_df.columns if i.lower().startswith('no_of')]]= full_df.loc[:, [i for i in full_df.columns if i.lower().startswith('no_of')]].fillna(0)


# In[227]:


full_df[[i for i in full_df.columns if i.lower().startswith('no_of')]].describe()


# In[228]:


full_df['AUM'].fillna(0, inplace = True)
full_df['AUM'] = full_df['AUM'].apply( lambda x : x if x > 0 else 0)
full_df['AUM'].describe()


# In[229]:


full_df.drop([col for col in full_df.columns if col.startswith('aum')], axis = 1 , inplace = True)
full_df.info()


# In[230]:


full_df.drop(['refresh_date_y', 'refresh_date_x'], axis = 1 , inplace = True)
full_df.info()


# In[231]:


full_df.iloc[:, 15:] = full_df.iloc[:, 15:].fillna(0)
full_df.info()


# In[232]:


firm_b = firm_b.rename({'Contact ID': 'CONTACT_ID'}, axis = 1)
full_df = pd.merge(full_df, firm_b, on = 'CONTACT_ID')


# In[233]:


full_df.drop([ 'CONTACT_ID'], axis = 1 , inplace = True)
full_df.info()


# In[234]:


full_df.groupby('Firm name')[['sales_12M', 'sales_curr', 'new_Fund_added_12M']].mean().nlargest(15,'sales_12M').plot(kind = 'barh')
full_df.groupby('Firm name')[['sales_12M', 'sales_curr', 'new_Fund_added_12M']].mean().nsmallest(15,'sales_12M').plot(kind = 'barh')


# In[235]:


def barcomparer(name):
    full_df.groupby(name)[['sales_12M', 'sales_curr', 'new_Fund_added_12M']].mean().nlargest(15,'sales_12M').plot(kind = 'barh')
    full_df.groupby(name)[['sales_12M', 'sales_curr', 'new_Fund_added_12M']].mean().nsmallest(15,'sales_12M').plot(kind = 'barh')


# In[215]:


barcomparer('Channel')


# In[236]:


barcomparer('Sub channel')


# In[292]:


def channel_maker(x):
    if x in ['IBD', 'NACS']:
        return x
    else:
        return 'other'


# In[313]:


full_df['Sub channel'] = full_df['Sub channel'].apply(channel_maker)
full_df = pd.get_dummies(full_df, columns = ['Sub channel'], drop_first = True)


# In[314]:


full_df.info()


# In[315]:


data = full_df
y = data['new_fund_2019']
y.value_counts()


# In[316]:


y = np.where(y >= 1,1,0)


# In[318]:


data.info()


# In[320]:


X = data.drop(['new_fund_2019', 'Sales_2019'], axis = 1)
X.info()


# ### Model Selection

# In[321]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)


# In[322]:


sscaler = StandardScaler()


# In[323]:


X_tr_sc = sscaler.fit_transform(X_train)
X_ts_sc = sscaler.fit_transform(X_test)


# In[324]:


lgr = LogisticRegression()
forest = RandomForestClassifier()
hist = HistGradientBoostingClassifier()


# In[325]:


lgr.fit(X_tr_sc, y_train)


# In[326]:


lgr.score(X_tr_sc, y_train)


# In[327]:


lgr.score(X_ts_sc, y_test)


# In[328]:


pd.value_counts(y_test, normalize = True)


# In[329]:


forest.fit(X_train, y_train)
forest.score(X_train, y_train)


# In[330]:


forest.score(X_test, y_test)


# In[331]:


hist.fit(X_train, y_train)
hist.score(X_train, y_train)


# In[332]:


hist.score(X_test, y_test)


# In[333]:


Forest_params = {'max_depth': [1,2,3,4,5],
                'min_samples_split': [2,3,4,5,6],
                'ccp_alpha': [0.001,0.01,.1,1.0]}


# In[334]:


forest_grid = GridSearchCV(forest, param_grid = Forest_params)


# In[335]:


forest_grid.fit(X_train, y_train)


# In[336]:


forest_grid.score(X_test, y_test)


# In[337]:


forest_grid.best_params_


# In[338]:


hist_params = {'max_depth': [1,2,3,4,5],
                'learning_rate': [0.1,0.01,1.0],
                'l2_regularization': [0.1,1.0,10,100, 1000]}


# In[339]:


hist_grid = GridSearchCV(hist, param_grid = hist_params, n_jobs = -1)


# In[340]:


hist_grid.fit(X_train, y_train)


# In[341]:


hist_grid.score(X_train, y_train)


# In[342]:


hist_grid.score(X_test, y_test)


# In[343]:


hist_grid.best_params_


# In[344]:


plot_confusion_matrix(hist_grid, X_test, y_test)


# In[345]:


plot_confusion_matrix(forest_grid, X_test, y_test)


# In[346]:


plot_confusion_matrix(lgr, X_test, y_test)


# ## Lift chart

# In[1144]:


data1a = data.loc[data['AUM']< 1000000]
data1b = data1a.loc[data1a['AUM']> 000]
data1c = data1b.loc[data1b['no_of_sales_12M_1']< 50]
data1 = data1c.loc[data1c['new_Fund_added_12M']< 1]


# In[1145]:


data1.info()


# In[1146]:


X = data1.drop(['new_fund_2019', 'Sales_2019'], axis = 1)
y = data1['new_fund_2019']


# In[1147]:


X = X[['no_of_sales_12M_1', 'AUM', 'sales_12M', 'new_Fund_added_12M']]
y = np.where(y>0,1,0)


# In[1148]:


X_train , X_test, y_train, y_test = train_test_split (X, y, stratify = y)


# In[1149]:


pipe = make_pipeline(PolynomialFeatures(), StandardScaler(), LogisticRegression())


# In[1150]:


params = {'polynomialfeatures__degree': [1,2],
         'logisticregression__C': [0.001,0.1,1.0,10.0]}


# In[1151]:


grid = GridSearchCV(pipe, param_grid = params)


# In[1152]:


grid.fit(X_train, y_train)


# In[1153]:


grid.score(X_train, y_train)


# In[1154]:


grid.score(X_test, y_test)


# In[1155]:


positive_probs = grid.predict_proba(X_test)[:, 1]
negative_probs = grid.predict_proba(X_test)[:, 0]


# In[1156]:


prob_df = pd.DataFrame({'prob': negative_probs, 'new_fund': y_test})


# In[1157]:


def lift_chart(probas, y, category = 'pos'):
    prob_df = pd.DataFrame({'prob': probas, 'y': y})
    if category  == 'neg':
        prob_df['y'] = np.where(prob_df['y']== 0, 1, 0)
    prob_df['decile'] = pd.qcut(prob_df['prob'], 10)
    lift_df = prob_df.groupby('decile')['y'].agg(['count', 'sum'])
    lift = lift_df['sum']/lift_df['sum'].sum()
    lift_df = pd.DataFrame({'lift' : lift, 'deciles': [10,9,8,7,6,5,4,3,2,1]})
    lifts = lift_df.sort_values('deciles')['lift'].cumsum().values
    lifts = np.insert(lifts, 0, 0)
    return lifts


# In[1158]:


prob_df.head()
prob_df['decile'] = pd.qcut(prob_df['prob'], 10)
prob_df


# In[1159]:


lift_df = prob_df.groupby('decile')['new_fund'].agg(['count', 'sum'])


# In[1160]:


lift = lift_df['sum']/lift_df['sum'].sum()


# In[1161]:


lift_df = pd.DataFrame({'lift': lift, 'deciles': [10,9,8,7,6,5,4,3,2,1]})
lifts = lift_df.sort_values('deciles')['lift'].cumsum().values
lifts = np.insert(lifts, 0, 0)


# In[1162]:


pos_lift = lift_chart(positive_probs , y_test)
neg_lift = lift_chart(negative_probs, y_test, 'neg')


# In[1163]:


x = [0, 0.1,.2,.3, .4, .5,.6,.7,.8,.9,1]
plt.plot(x,x, '--o')
plt.plot(x, lifts, label = 'Lift Curve for adding a new fund')
plt.plot(x, neg_lift, label = 'Lift Curve for not adding a new fund')
plt.legend();


# In[1164]:


probas = grid.predict_proba(X_test)


# In[1165]:


skplot.metrics.plot_cumulative_gain(y_test, probas)


# In[1166]:


lift_df = pd.DataFrame({'baseline': x, 'lift' : lift_chart(probas[:,1], y_test)})


# In[1167]:


lift_df['lift_over_average'] = lift_df['lift'] - lift_df['baseline']
lift_df


# In[1168]:


from sklearn.preprocessing import PowerTransformer
ptransformer = PowerTransformer()
X_train_transformed = ptransformer.fit_transform(X_train)
fig, ax = plt.subplots(4,2, figsize = (14,16))
for i in range(4):
    ax[i,0].hist(X_train_transformed[:, i], label = 'Transformed')
    ax[i,1].hist(X_train.iloc[:, i], label = 'Original')
    ax[i,0].legend()
    ax[i,1].legend()
    ax[i, 0].set_title(f'{X_train.columns[i]}')


# In[1261]:


prob_df.index = X_test.index
whole_df = pd.concat((X_test, prob_df), axis = 1)


# In[1262]:


def thous(x, pos):
    return '${:1.1f}k'.format(x*1e-3)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
y=  sorted(whole_df.groupby('decile')['sales_12M'].mean())
x_val = [1,2,3,4,5,6,7,8,9,10]
y_val = y
ax.bar(x_val,y_val)
ax.yaxis.set_major_formatter(thous)
plt.title('$ Sales in 2018 ')
plt.ylabel('$ Thousands')
plt.xlabel('Deciles')
plt.show()


# In[1263]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
y=  sorted(whole_df.groupby('decile')['no_of_sales_12M_1'].mean())
x_val = [1,2,3,4,5,6,7,8,9,10]
y_val = y
ax.bar(x_val,y_val)
plt.title('Number of Sales in 2018 ')
plt.ylabel('# of Sales')
plt.show()


# In[1171]:


from matplotlib import pyplot


# In[1174]:


lgr.fit(X_train, y_train)


# In[1241]:


plot_confusion_matrix(lgr, X, y)


# In[1240]:


plot_confusion_matrix(grid, X_test, y_test)


# In[ ]:




