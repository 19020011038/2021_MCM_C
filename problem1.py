import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from math import radians, cos, sin, asin, sqrt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


# read the file
df = pd.read_excel('2021MCMProblemC_DataSet.xlsx')

# select the right data
reports = df.loc[(df['Lab Status'] == 'Positive ID'), ['Detection Date', 'Latitude', 'Longitude']]
ndata = np.array(reports)
reportsList = ndata.tolist()
reportsList = sorted(reportsList, key=lambda s: s[0])

# make time series list
distanceList = []
dateList = []
for index in reportsList:
    distanceList.append(haversine(reportsList[0][2], reportsList[0][1], index[2], index[1]))
    dateList.append(index[0])
distanceList.remove(distanceList[0])
dateList.remove(dateList[0])

dateList_int = []
for index in dateList:
    dateList_int.append(int(((index - dateList[0]).total_seconds() / (24 * 60 * 60))))
print('The days of data: ')
print(dateList_int)

num_list = []
dis_list = []
dis_index = 0
for i in range(dateList_int[len(dateList_int) - 1] + 1):
    num_list.append(i)
    if i in dateList_int:
        dis_list.append(distanceList[dis_index])
        dis_index += 1
    else:
        dis_list.append(np.nan)

temp_d = {'num': num_list, 'dis': dis_list}
df_set = DataFrame(temp_d)
df_set = df_set.interpolate()

x_list = np.array(df_set['num']).tolist()
y_list = np.array(df_set['dis']).tolist()

# original picture: distance-date
dta = pd.Series(y_list)
dta.index = x_list
dta.plot()
plt.title('')
plt.xlabel('Days')
plt.ylabel('Distance')


# ADF
result = adfuller(y_list)
print('The result of ADF before difference: ')
print(result)

# ACF
plot_acf(dta)

# differences
diff1 = dta.diff(1).dropna()
diff1.columns = ['values']
diff1.plot()


print('The result of ADF after difference: ')
print(adfuller(diff1))

# white noise detection
print(u'The result of white noise detection：', acorr_ljungbox(diff1, lags=1))

# ACF2
plot_acf(diff1)

# PACF
plot_pacf(diff1)

# MODEL
model = sm.tsa.ARIMA(dta, order=(4, 1, 2))
results = model.fit(disp=0)
print('The detail of the model:')
print(results.summary())

# predict
predict_sunspots = results.predict(start=1, end=367)
predict_sunspots_2 = results.forecast(100)
print('The result of the prediction, next 100 :')
print(predict_sunspots_2[0])

# draw predictions
fig, ax = plt.subplots(figsize=(8, 4))
ax = diff1.plot(ax=ax)
predict_sunspots.plot(ax=ax)


# Model evaluation
residuals = pd.DataFrame(results.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
print(residuals)

s = pd.DataFrame(results.resid, columns=['value'])
u = s['value'].mean()  # calculate the average
print(u)
std = s['value'].std()  # calculate the standard deviation
print(std)

# show results
plt.show()
