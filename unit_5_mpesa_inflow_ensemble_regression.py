# Unit 5: M-PESA inflow ensemble regression
# Extracted from Appendix A of Team_FloppyDisk_Extended_Final_Report.docx.
# For the original Colab notebook where available, see the matching .ipynb file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from IPython.display import display
except ImportError:
    display = print

CGAP_URL = 'https://www.cgap.org/sites/default/files/2019-08/Busara--CGAP-%20Understanding%20Use%20of%20Financial%20Services_Dataset.xlsx'
WFP_URL = 'https://data.humdata.org/dataset/e0d3fba6-f9a2-45d7-b949-140c455197ff/resource/517ee1bf-2437-4f8c-aa1b-cb9925b9d437/download/wfp_food_prices_ken.csv'
# Cached copies keep local reruns fast; Colab downloads if absent.
cache = Path('unit5_build') if Path('unit5_build').is_dir() else Path('.')
cgap_file, wfp_file = cache/'cgap_mpesa.xlsx', cache/'wfp_kenya.csv'
if not cgap_file.exists(): urlretrieve(CGAP_URL, cgap_file)
if not wfp_file.exists(): urlretrieve(WFP_URL, wfp_file)
SEED=42

raw=pd.read_excel(cgap_file, usecols=['sid','completion','status','paid_in','withdrawn'])
raw['completion']=pd.to_datetime(raw['completion'],errors='coerce')
raw=raw[(raw['status']=='Completed') & raw['sid'].notna() & raw['completion'].between('2018-02-01','2018-07-31 23:59:59')].copy()
raw['month']=raw['completion'].dt.to_period('M').dt.to_timestamp()
raw['paid_in']=pd.to_numeric(raw['paid_in'],errors='coerce').fillna(0)
raw['outflow']=-pd.to_numeric(raw['withdrawn'],errors='coerce').fillna(0)
monthly=raw.groupby(['sid','month'],as_index=False).agg(inflow=('paid_in','sum'),outflow=('outflow','sum'),txn_count=('completion','size'))
print('Completed transactions:',len(raw),'| Accounts:',raw.sid.nunique(),'| Active account-months:',len(monthly))

prices=pd.read_csv(wfp_file)
prices['date']=pd.to_datetime(prices['date'],errors='coerce')
prices=prices[(prices['market'].str.casefold()=='nairobi') & (prices['priceflag']=='actual') & (prices['pricetype']=='Retail') & (prices['currency']=='KES')].copy()
wanted={('Bread','400 G'):'bread',('Milk (cow, pasteurized)','500 ML'):'milk',('Oil (vegetable)','L'):'oil'}
prices=prices[prices.apply(lambda r:(r['commodity'],r['unit']) in wanted,axis=1)].copy()
prices['key']=prices.apply(lambda r:wanted[(r['commodity'],r['unit'])],axis=1)
prices['month']=prices['date'].dt.to_period('M').dt.to_timestamp()
prices['price']=pd.to_numeric(prices['price'],errors='coerce')
wide=prices.pivot_table(index='month',columns='key',values='price',aggfunc='mean').sort_index()
for col in ['bread','milk','oil']:
    wide[col+'_change_pct']=100*wide[col].pct_change(fill_method=None)
wide=wide.reset_index()
data=monthly.merge(wide,on='month',how='left',validate='many_to_one').sort_values(['sid','month'])
next_rows=monthly[['sid','month','inflow']].rename(columns={'inflow':'next_inflow'}).copy()
next_rows['month']=next_rows['month']-pd.offsets.MonthBegin(1)
data=data.merge(next_rows,on=['sid','month'],how='left',validate='one_to_one')
features=['inflow','outflow','txn_count','bread','milk','oil','bread_change_pct','milk_change_pct','oil_change_pct']
data=data.dropna(subset=features+['next_inflow']).copy()
train=data[data.month.between('2018-02-01','2018-04-01')].copy()
test=data[data.month==pd.Timestamp('2018-06-01')].copy()
assert len(train)>0 and len(test)>0 and train.month.max()+pd.DateOffset(months=1)<test.month.min()
print('Train feature months:',train.month.min().date(),'to',train.month.max().date(),'| rows:',len(train))
print('Gap: May 2018 | Test features: June 2018 | Target: July 2018 | rows:',len(test))
print('Train accounts:',train.sid.nunique(),'| Test accounts:',test.sid.nunique(),'| overlap:',len(set(train.sid)&set(test.sid)))

models={'Mean baseline':DummyRegressor(strategy='mean'),
 'CART regression tree':DecisionTreeRegressor(max_depth=5,min_samples_leaf=12,random_state=SEED),
 'Random Forest':RandomForestRegressor(n_estimators=150,max_depth=9,min_samples_leaf=8,n_jobs=-1,random_state=SEED),
 'Gradient Boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=12,l2_regularization=2,random_state=SEED)}
results=[]
for name,model in models.items():
    model.fit(train[features],np.log1p(train['next_inflow']))
    pred=np.maximum(0,np.expm1(model.predict(test[features])))
    results.append({'model':name,'mae_kes':mean_absolute_error(test['next_inflow'],pred),
                    'rmse_kes':np.sqrt(mean_squared_error(test['next_inflow'],pred))})
results=pd.DataFrame(results).sort_values('mae_kes').reset_index(drop=True)
display(results.round(2))

ax=results.set_index('model')['mae_kes'].sort_values().plot.barh(color='#1767ff',figsize=(8,4))
ax.invert_yaxis();ax.set(xlabel='MAE (KES per active account-month)',title='June features → July inflow: held-out test')
plt.tight_layout();plt.show()
