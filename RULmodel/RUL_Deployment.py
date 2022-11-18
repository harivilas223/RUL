#!/usr/bin/env python
# coding: utf-8

# In[145]:


# pip install wtte

#import required libraries
import numpy as np 
import tensorflow
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import BatchNormalization
#import tensorflow as tf
tensorflow.config.run_functions_eagerly(True)
from keras import backend as K
from tensorflow.keras import callbacks
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from six.moves import xrange
import numpy as np
import matplotlib.pyplot as plt
#import keras
#from keras.models import Sequential
#from keras.layers import Dense

#from keras.layers import LSTM,GRU
#from keras.layers import Lambda
#from keras.layers.wrappers import TimeDistributed

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import History

import wtte.weibull as weibull
#import wtte.wtte as wtte

from wtte.wtte import WeightWatcher

np.random.seed(2)
pd.set_option("display.max_rows",1000)

#from wtte.wtte import output_lambda
import wtte
#from wtte.wtte import WeightWatcher
import tensorflow as tf
from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tqdm
from tqdm import tqdm
######connect,import and read the data from data base
# import pandas as pd
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from google.cloud import firestore
import google.cloud
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
from PIL import Image
from google.cloud import storage
from datetime import datetime, timedelta,date
cred = credentials.Certificate(r"/home/srinivas/SONA_BLW/sonaConfig.json")
firebase_admin.initialize_app(cred)
import datetime as dt
import requests
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pdcolumn_list
import json
from pandas.io.json import json_normalize
from datetime import datetime,timedelta,date
def bigtable_data_downloader(asset, customer, start_dt, end_dt):

    analysis_start_dt = start_dt + timedelta(hours=5, minutes=30)
    analysis_end_dt = end_dt + timedelta(hours=5, minutes=30)
    print('Downloading data for '+ asset +' from ' + str(analysis_start_dt) + ' to ' + str(analysis_end_dt))

    req_df = pd.DataFrame()
    start_time = start_dt
    while(start_time < end_dt):
        end_time = start_time + timedelta(minutes=10)

        unix_from = str(int(start_time.timestamp()))
        unix_to = str(int(end_time.timestamp()))
        print('start', unix_from)
        print('end', unix_to)

        asset_code = '_'.join(asset.split(' '))
        print(asset_code)

        request_body = {"customerId": customer, "assetcode": asset_code, "startTime": unix_from,
                        "endTime": unix_to}
        history_data = requests.post('https://datastore.orionintelligentenergy.com/api/bigtable/select', json=request_body)
        print(history_data)

        result_df = pd.DataFrame()
        metadata = {}
        tags = []
        if history_data.status_code == 200:
            history_data = history_data.json().get('data')
            print(len(history_data))
            if len(history_data) != 0:
                metadata = history_data[0]
                nested_data = metadata.get('tags')
                inner_temp = json_normalize(nested_data)
                tags = inner_temp['tag'].unique()
                del metadata['tags']

                total_data = []

                for each_data in history_data:
                    if 'tags' in each_data.keys():
                        tag_data = each_data.get('tags')
                        del each_data['tags']
                        # if len(tag_data) > 2:
                        for data in tag_data:
                            each_data[data['tag']] = data['value']

                        total_data.append(each_data)

                result_df = pd.DataFrame(total_data)
        if not result_df.empty:
            result_df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        print("Data downloaded " + str(result_df.shape) + ' records')

        start_time = end_time
        req_df = req_df.append(result_df, ignore_index=True, sort=False)

    return req_df
#bd_1= pd.read_csv("/home/srinivas/SONA_BLW/data.csv")
def data_procesing():
    today = datetime.today()
    target_date = today - timedelta(days=1)
    start_time = target_date
    end_time = today
    #start_dt=datetime.strptime('05-08-2021 05:30:00', '%d-%m-%Y %H:%M:%S')
    #end_dt=datetime.strptime('06-08-2021 05:30:00', '%d-%m-%Y %H:%M:%S')
    #bd_2 = bigtable_data_downloader("SONA01UNIT01ASST01RGH01", 'SON',start_time,end_time)

    #Data preprocesing 
    bd_2= pd.read_csv("/home/srinivas/SONA_BLW/data1.csv")
    bd_2=bd_2[['timestamp','AB_FLYWHEEL_GAP_ASCENT']]
    #bd_1=pd.concat([bd_3,bd_2[['timestamp','AB_FLYWHEEL_GAP_ASCENT']]])
    bd_1=bd_2.reset_index(drop=True)
    #bd_1=bd_2[['timestamp','AB_FLYWHEEL_GAP_ASCENT']]
    bd_1.to_csv('/home/srinivas/SONA_BLW/data1.csv')
    bd_1['timestamp']=bd_1['timestamp'].values.astype('<M8[m]')
    #bd_1=bd_1[bd_1['timestamp'].between('2021-07-18 00:00:00','2021-07-19 23:59:00')]
    bd_1['unit_number']=1
    days_counter = 0

    for i in range(len(bd_1)):    
        # if Ril_data_2.loc[i,'counter2']!= 0:
            days_counter = days_counter+1
            bd_1.at[i,'days']=days_counter

    #########################

    col2 = ['AB_FLYWHEEL_GAP_ASCENT']
    id_col = 'unit_number'
    time_col = 'days'
    column_names = [id_col, time_col] + col2
    unit_data_copy = bd_1[column_names]
    unit_data_copy=unit_data_copy.fillna(method='bfill')
    scaler=pipeline.Pipeline(steps=[
    #     ('z-scale', StandardScaler()),
         ('minmax', MinMaxScaler(feature_range=(-1, 1))),
         ('remove_constant', VarianceThreshold())
    ])
    test2=unit_data_copy[unit_data_copy['unit_number'].isin([1.0])]
    test2 = np.concatenate([test2[['unit_number', 'days']], scaler.fit_transform(test2[col2])], axis=1)
    test2[:, 0:2] -= 1
    return test2

#convert the datashape into LSTM input sahpe
# TODO: replace using wtte data pipeline routine
def build_data(engine, time, x, max_time, is_test, mask_value):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = []
    
    # number of features
    d = x.shape[1]

    # A full history of sensor readings to date for each x
    out_x = []
    list1=[0]
    for i in tqdm(list1):
#     n_engines=5
#     for i in tqdm(range(n_engines)):
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y.append(np.array((max_engine_time - j, 1), ndmin=2))

            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value
#             xtemp = np.full((1, max_time, d), mask_value)
            
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x.append(xtemp)
            
        this_x = np.concatenate(this_x)
        out_x.append(this_x)
    out_x = np.concatenate(out_x)
    out_y = np.concatenate(out_y)
    return out_x, out_y

# test2=data_procesing()
 ########################################
def model_data(): 
    test2=data_procesing()
    max_time = 100
    mask_value = -99
    test_x2,_= build_data(engine=test2[:, 0], time=test2[:, 1], x=test2[:, 2:], max_time=max_time, is_test=True, mask_value=mask_value)
    test_y1 = [0]
    test_y1 = pd.DataFrame(test_y1,columns=['T'])
    test_y1= test_y1.copy()
    test_y1['E'] = 1
    test_x2=test_x2.astype(np.float32)
    test_y1=test_y1.astype(np.float32)
    return test_x2,test_y1
#reload the saved model.


#make the predection from loaded model
def prediction():
    model = tensorflow.keras.models.load_model('/home/srinivas/SONA_BLW/RULmodel',compile=False)
    test_x2,test_y1=model_data()
    test_predict = model.predict(test_x2)
    test_predict = np.resize(test_predict, (1, 2))
    test_result = np.concatenate((test_y1, test_predict), axis=1)
    test_results_df1 = pd.DataFrame(test_result, columns=['T', 'E', 'alpha', 'beta'])
    test_results_df1['unit_number'] = np.arange(1, test_results_df1.shape[0]+1)
    return test_results_df1
# test_results_df = pd.DataFrame(test_result, columns=['T', 'E', 'alpha', 'beta'])
# test_results_df['unit_number'] = np.arange(1, test_results_df.shape[0]+1)


#write the weibull formulas
import math
def weibull_pdf(alpha, beta, t):
    return (beta/alpha) * (t/alpha)**(beta-1)*np.exp(- (t/alpha)**beta)
def weibull_median(alpha, beta):
    return alpha*(-np.log(.5))**(1/beta)
def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)
def weibull_mode(alpha, beta):
    assert np.all(beta > 1)
    return alpha * ((beta-1)/beta)**(1/beta)

########################
def mean_median_mode():
    test_results_df1=prediction()
    test_results_df1['predicted_mu'] = test_results_df1[['alpha', 'beta']].apply(lambda row: weibull_mean(row[0], row[1]), axis=1)
    test_results_df1['predicted_median'] = test_results_df1[['alpha', 'beta']].apply(lambda row: weibull_median(row[0], row[1]), axis=1)
    test_results_df1['predicted_mode'] = test_results_df1[['alpha', 'beta']].apply(lambda row: weibull_mode(row[0], row[1]), axis=1)
    return test_results_df1
#write the weibull reliability formula
def reliability(alpha, beta,t):
    return np.exp(- (t/alpha)**beta)
############################################

test_results_df=mean_median_mode()
sdate = date(2022,10,17)   
edate = date(2022,11,17)
list1=pd.date_range(sdate,edate-timedelta(days=1),freq='d')
def plot_weibull_predictions(results_df):
    palette = sns.color_palette("RdBu_r", results_df.shape[0]+1)
    color_dict = dict(enumerate(palette))
    for i, row in enumerate(results_df.iterrows()):
            alpha=row[1]['alpha']
            beta = row[1]['beta']
            T = row[1]['T']
            label = 'a={} b={}'.format(alpha, beta)
            color = color_dict[i]
            t=np.arange(0,31)
            fig = plt.figure(figsize=(10, 5), dpi=80)
            ax1 = fig.add_subplot(111)
            fig.autofmt_xdate()
            alpha=(alpha*5)/(60*1440)
            ax1.plot(list1, reliability(alpha, beta, t), color=color, label=label)
            interp_func1 = scipy.interpolate.interp1d(t,reliability(alpha, beta, t))
            interp_func = scipy.interpolate.interp1d(reliability(alpha, beta, t),t)
	    #threshold=interp_func(0.2)
            #ax1.scatter(x=threshold,y=0.2, color='r', linestyle='--')
            ax1.set_xlabel("TIME",fontweight ='bold')
            ax1.set_ylabel("RELIABILITY",fontweight ='bold')
            RUL = interp_func(0.02)
            current_r=interp_func1((date.today()-date(2022, 10, 17)).days)
            #print('Duraion of time ramined to reach at 0.2 relaibity is',threshold,'days')
            #print('Remaing useful time at 0.2 relaibity is',RUL,'days')
#             pd.to_datetime(date.today(), format='%Y%m%d', errors='ignore')
            figure1=plt.savefig('/home/srinivas/SONA_BLW/figure1.png')
            figure=str(datetime.now().strftime('%Y-%m-%d'))+'.'+'png'
            credentials = service_account.Credentials.from_service_account_file(r"/home/srinivas/SONA_BLW/sonaConfig.json")
            project_id = 'Sona-BLW'
            client = storage.Client(credentials= credentials,project=project_id)
            bucket = client.get_bucket('docile_images')
            blob = bucket.blob('Sona_DIscliner_Relability'+'/'+figure)
            blob.upload_from_filename('/home/srinivas/SONA_BLW/figure1.png',content_type='image/jpeg')
            print('image uploaded succesfully')
            Discliner_RUL_json={
            "customer":"SONA",
            "asset":"Rough Press",
            "asset code":"SONA01UNIT01ASST01RGH01",
            "component":"Disc liner",
            "sensor":"Ascent Gap",
            "Date":datetime.now(),
            "Threshold":float(interp_func(0.2)),
            "Zero_reliability":float(RUL),
            "current_reliability":float(current_r),
            "liner change date":'2022-10-17',
            "img_path":blob.public_url
             }  
            db = firestore.client()
            db.collection(u'Asset_OEE').document(u'SONA').collection(u'analytics').document().set(Discliner_RUL_json)
            print('file uploaded')
#           #db.collection(u'Asset_OEE').document(u'SONA').collection(u'DISC_liner_RUL.img').document().set(figure1)
            # use pillow to open and transform the file
            #image = Image.open('/home/srinivas/SONA_BLW/figure1.png')
            # perform transforms
            #image.save('/home/srinivas/SONA_BLW/figure1.png')
            #of = open('/home/srinivas/SONA_BLW/figure1.png', 'rb')
            #blob.upload_from_file(of)
            # or... (no need to use pillow if you're not transforming)
            #blob.upload_from_filename(filename='/home/srinivas/SONA_BLW/figure1.png')
    plt.show()
 ###################
plot_weibull_predictions(test_results_df)

