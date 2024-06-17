#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np

import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# with open('model2.bin', 'rb') as f_in:
#     dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


month = int(sys.argv[1])
year = int(sys.argv[2])

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')




# df.head()




dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print("The mean predicted duriation is ", np.mean(y_pred))


df['ride_id'] = f'{year:04}/{month:02}_' + df.index.astype('str') 


# df_result = df[['ride_id', 'predictions']]
df_result = pd.DataFrame({
    "ride_id": df["ride_id"],
    "predictions": y_pred
})
output_file = f'df_result_{month:02d}_{year:04d}.parquet'
bucket_name = 'mlflow-1234'
output_path = f'gs://{bucket_name}/output/{output_file}'

df_result.to_parquet(
    output_path,
    engine='pyarrow',
    compression=None,
    index=False
)

# df_loaded = pd.read_parquet(ouput_file, engine='pyarrow')
df_loaded = pd.read_parquet(output_path, engine='pyarrow')

# Print the loaded DataFrame to verify
print(df_loaded.head())






# get_ipython().system("jupyter nbconvert starter.ipynb -to 'script'")


# # In[32]:


# nbconvert


# # In[ ]:




