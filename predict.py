import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("sphist.csv")
print(df.shape)

df['Date'] =  pd.to_datetime(df['Date'])
#df = df[ df['Date'] > datetime(year=2015, month=4, day=1) ]
df.sort_values('Date',ascending=True)

for i in range(len(df)):
    #print(data.iloc[i]['Date'])
    if (i >= 3):
        df.loc[i,'day_3'] = (df.loc[i-1,'Close']+df.loc[i-2,'Close']+df.loc[i-3,'Close'])/3
    else:
        df.loc[i,'day_3'] = 0
        
df['day_5'] = pd.rolling_mean(df['Close'], window = 5).shift(1)
df['day_30'] = pd.rolling_mean(df['Close'], window = 30).shift(1)
df['day_365'] =  pd.rolling_mean(df['Close'], window = 365).shift(1)

df['ratio_dy'] = df['day_5']/df['day_365']
df['AvVol_5day'] = pd.rolling_mean(df['Volume'], window = 5).shift(1)
df['AvVol_1yr'] = pd.rolling_mean(df['Volume'], window = 365).shift(1)
        
df = df[ df['Date'] > datetime(year=1951, month=1, day=2)]
df = df.dropna(axis=0)



train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] > datetime(year=2013, month=1, day=1)]

feature = ['day_5','day_30','day_365']
target = 'Close'

lr = LinearRegression()
lr.fit(train[feature],train['Close'])

test_prediction = lr.predict(test[feature])

mae = sum(abs(test_prediction - test['Close'])) / len(test_prediction)
print(mae)





        

