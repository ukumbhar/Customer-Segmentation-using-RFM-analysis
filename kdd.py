import numpy as np 
import pandas as pd
from pandasql import sqldf
import seaborn as sns
import datetime as dt

from sklearn.metrics import silhouette_score

from scipy import stats

import matplotlib.pyplot as plt

articles = pd.read_csv('articles.csv')
customers = pd.read_csv('customers.csv')
transactions = pd.read_csv('transactions_train.csv')
df = transactions.copy()
df['date'] = pd.to_datetime(df['t_dat']).dt.date
df.drop(columns=['t_dat'], inplace=True)
df.head()


NOW = dt.date(2020, 9, 22)

rfmTable = df.groupby('customer_id').agg({'date': lambda x: (NOW - x.max()).days,
                                                'article_id': lambda x: len(x),
                                                'price' : lambda x: round(x.sum(), 2)})

rfmTable["date"] = rfmTable["date"].astype(int)
rfmTable.rename(columns={'date': 'Recency',
                         'article_id': 'Frequency', 
                         'price': 'Monetary'}, inplace=True)
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
rfmSegmentation = rfmTable.copy()
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
rfmSegmentation['R_Quartile'] = rfmSegmentation['Recency'].apply(RClass, args=('Recency', quantiles))
rfmSegmentation['F_Quartile'] = rfmSegmentation['Frequency'].apply(FMClass, args=('Frequency', quantiles))
rfmSegmentation['M_Quartile'] = rfmSegmentation['Monetary'].apply(FMClass, args=('Monetary', quantiles))

rfmSegmentation['RFM_Scores'] = rfmSegmentation.R_Quartile.apply(str) \
                            + rfmSegmentation.F_Quartile.apply(str) \
                            + rfmSegmentation.M_Quartile.apply(str)

label = list(np.zeros(len(rfmSegmentation)))

for i in range(len(rfmSegmentation)):
    if rfmSegmentation['RFM_Scores'].iloc[i] =='444':
      label[i] = "Best_Customers"  
    elif rfmSegmentation['RFM_Scores'].iloc[i][1]=='4':
      label[i] = "Loyal_Customers"
    elif rfmSegmentation['RFM_Scores'].iloc[i][2]=='4': 
      label[i] = "Big Spenders"
    elif rfmSegmentation['RFM_Scores'].iloc[i]=='244' : 
      label[i] = "Almost Lost"
    elif rfmSegmentation['RFM_Scores'].iloc[i]=='144' : 
      label[i] = "Lost Customers"
    elif rfmSegmentation['RFM_Scores'].iloc[i] =='111' : 
      label[i] = "Lost Thrifty Customers"

Loyal_Customers=['433']
Big_Spenders=['413','432','423']
Almost_Lost=['333','233','422','331','313','431','323']
Lost_Customers=['222','311','322','223','332','411','133','132','312','131','123','213','113','421','412','231','321','232']
Lost_Cheap_Customers=['122','211','112','121','212','221']

for i in range(len(rfmSegmentation)):
    if rfmSegmentation['RFM_Scores'].iloc[i] in Loyal_Customers : 
      label[i] = "Loyal_Customers"
    elif rfmSegmentation['RFM_Scores'].iloc[i] in Big_Spenders : 
      label[i] = "Big Spenders"
    elif rfmSegmentation['RFM_Scores'].iloc[i] in Almost_Lost : 
      label[i] = "Almost Lost"
    elif rfmSegmentation['RFM_Scores'].iloc[i] in Lost_Customers : 
      label[i] = "Lost Customers"
    elif rfmSegmentation['RFM_Scores'].iloc[i] in Lost_Cheap_Customers  : 
      label[i] = "Lost Thrifty Customers"

rfmSegmentation['RFM_Scores_Segments'] = label
rfmSegmentation=rfmSegmentation.reset_index()

rfmSegmentation['RFM_Points'] = rfmSegmentation[['R_Quartile', 'F_Quartile', 'M_Quartile']].sum(axis=1).astype('float')

label = list(np.zeros(len(rfmSegmentation)))

for i in range(len(rfmSegmentation)):
    if rfmSegmentation['RFM_Points'].iloc[i] ==12 :
      label[i] = "Best_Customers"
    elif rfmSegmentation['RFM_Points'].iloc[i] ==11 :
      label[i] = "Loyal_Customers"
    elif rfmSegmentation['RFM_Points'].iloc[i] >= 9 :
      label[i] = "Big_Spenders"
    elif rfmSegmentation['RFM_Points'].iloc[i] >= 7 :
      label[i] = "Almost_Lost"
    elif rfmSegmentation['RFM_Points'].iloc[i] >= 5 :
      label[i] = "Lost_Customers"
    else : label[i] = "Lost_Thrifty_Customers"

label = list(np.zeros(len(rfmSegmentation)))

for i in range(len(rfmSegmentation)):
    if rfmSegmentation['RFM_Points'].iloc[i] ==12 :
      label[i] = "Best_Customers"
    elif rfmSegmentation['RFM_Points'].iloc[i] ==11 :
      label[i] = "Loyal_Customers"
    elif rfmSegmentation['RFM_Points'].iloc[i] >= 9 :
      label[i] = "Big_Spenders"
    elif rfmSegmentation['RFM_Points'].iloc[i] >= 7 :
      label[i] = "Almost_Lost"
    elif rfmSegmentation['RFM_Points'].iloc[i] >= 5 :
      label[i] = "Lost_Customers"
    else : label[i] = "Lost_Thrifty_Customers"

rfmSegmentation['RFM_Points_Segments'] = label


rfm = rfmSegmentation[['customer_id','Recency','Frequency','Monetary']]

# Set the Numbers
rfm_fix = pd.DataFrame()
rfm_fix["Recency"] = pd.Series(np.cbrt(rfm['Recency'])).values
rfm_fix["Frequency"] = stats.boxcox(rfm['Frequency'])[0]
rfm_fix["Monetary"] = pd.Series(np.cbrt(rfm['Monetary'])).values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rfm_fix)
rfm_normalized = scaler.transform(rfm_fix)
print(rfm_normalized.mean(axis = 0).round(2))
print(rfm_normalized.std(axis = 0).round(2))

from sklearn.cluster import KMeans
kmeans = KMeans()

kmeans = KMeans(n_clusters = 3).fit(rfm_normalized)
labels = kmeans.labels_
rfm['Kmeans_Label_ID']=labels
label1=kmeans.predict(rfm_normalized)
print(f'Silhouette Score(n=3): {silhouette_score(rfm_normalized, label1)}')



k_means2 = KMeans(n_clusters=2)
k_means2.fit(rfm_normalized)
labels2 = k_means2.labels_
rfm['Kmeans_Label_ID2']=labels2
label2=k_means2.predict(rfm_normalized)
print(f'Silhouette Score(n=2): {silhouette_score(rfm_normalized, label2)}')



























