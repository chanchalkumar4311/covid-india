import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateLocator
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

df = pd.read_csv('complete (1).csv', index_col='Date', parse_dates=True)
df.head()

df[['Total Confirmed cases (Indian National)','Total Confirmed cases ( Foreign National )','Cured/Discharged/Migrated','Death']] = df[['Total Confirmed cases (Indian National)','Total Confirmed cases ( Foreign National )','Cured/Discharged/Migrated','Death']].fillna(0).astype(int)

df['Day Number'] = range(1,df.shape[0]+1)

df['Confirmed'] = df['Total Confirmed cases (Indian National)'] + df['Total Confirmed cases ( Foreign National )']

df[['Confirmed','Cured/Discharged/Migrated', 'Death']].plot(color=['b','g','r'])

plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Cases")
plt.title("Corona virus in India")
plt.legend()
plt.grid(False)
plt.show()

df[['Confirmed','Cured/Discharged/Migrated', 'Death']].plot(color=['b','g','r'])
plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Cases")
plt.title("Corona virus in India")
plt.legend()
plt.grid(False)
plt.show()
covid_state_cases_india = pd.DataFrame(df.groupby('Name of State / UT',as_index=True).sum()['Confirmed'])
covid_state_cases_india.reset_index(inplace=True)
covid_state_cases_india.columns= ['Name of State / UT','Confirmed']
covid_state_cases_india.head()

fig, ax1 = plt.subplots(figsize=(17, 8))
g=sns.barplot(x="Name of State / UT", y="Confirmed",data = covid_state_cases_india)
plt.xticks(rotation=75)
plt.ylabel('Cumulatve Corona Cases In India')
plt.xlabel('Date')
plt.title('Cumulative Corona Cases in Inida',fontsize = 25, weight = 'bold')
plt.show()
covid_daily_cases_india = pd.DataFrame(df.groupby('Date',as_index=True).sum()['Confirmed'])
covid_daily_cases_india.reset_index(inplace=True)
covid_daily_cases_india.columns= ['Date','Confirmed']
covid_daily_cases_india.head()
fig, ax1 = plt.subplots(figsize=(17, 8))
g=sns.barplot(x="Date", y="Confirmed",data = covid_daily_cases_india)
plt.xticks(rotation=75)
plt.ylabel('Cumulatve Corona Cases In India')
plt.xlabel('Date')
plt.title('Cumulative Corona Cases in Inida',fontsize = 25, weight = 'bold')
plt.show()

df['Active'] = df['Confirmed'] - df['Cured/Discharged/Migrated'] - df['Death']
modal_IND = LinearRegression(fit_intercept=True)
poly = PolynomialFeatures(degree=8)
num_days_poly = poly.fit_transform(df['Day Number'].to_numpy().reshape(-1,1))
poly_reg = modal_IND.fit(num_days_poly, df['Confirmed'].to_numpy().reshape(-1,1))
predictions_for_given_days = modal_IND.predict(num_days_poly)

plt.style.use('fivethirtyeight')
plt.plot(df['Day Number'], df['Confirmed'],color='b',)
plt.plot(df['Day Number'], df['Cured/Discharged/Migrated'],color='g')
plt.plot(df['Day Number'], df['Death'],color='r')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Corona virus in India",)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

tomorrow_value = df["Day Number"].iloc[-1] + 1 
value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))
prediction = modal_IND.predict(value_prediction)
print(f'Prediction for tomorrow (for day number {tomorrow_value}) : {prediction} cases ')

confirmed_cases = df['Confirmed'].to_numpy()
added_cases = np.diff(confirmed_cases)
added_cases_no_day = range(1,added_cases.shape[0]+1)
df['Added Cases'] = 0
df['Added Cases'][1:] = added_cases
added_cases_prediction = prediction[0][0] - df['Confirmed'][-1:]
print(f'Hospitals should be ready with {round(added_cases_prediction[:1].iloc[-1])} new beds.')
df[['Added Cases']].tail()

plt.style.use('fivethirtyeight')
df[['Added Cases']].plot(color='b')

plt.xlabel("Date")
plt.ylabel("New Added Cases")
plt.title("Corona virus in India",)
plt.legend()
plt.grid(True)
plt.show()

df_temp2=pd.DataFrame()
df_temp2['Confirmed'] = df['Confirmed']
df_temp2['Deaths'] = df['Death']
df_temp2['Recovered'] = df['Cured/Discharged/Migrated']
sns.pairplot(df_temp2)