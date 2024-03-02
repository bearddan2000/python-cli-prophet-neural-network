import pandas as pd
from prophet import Prophet

df = pd.read_csv('data.csv')

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=1)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print(forecast)

# print(forecast[:1]['trend'])