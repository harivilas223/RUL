import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime, timedelta
import requests
from pandas.io.json import json_normalize
import joblib
#from sklearn.externals import joblib
from keras.models import Sequential


cred = credentials.Certificate(r"/home/srinivas/SONA_BLW/sonaConfig.json")
firebase_admin.initialize_app(cred)


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


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

customer = 'SON'
assetcode = "SONA01UNIT01ASST01RGH01"

today = datetime.today()
target_date = today - timedelta(days=1)
start_time = target_date.strftime("%d-%m-%Y") + " 00:00:00"
end_time = today.strftime("%d-%m-%Y") + " 00:00:00"
download_start_dt = datetime.strptime(start_time, '%d-%m-%Y %H:%M:%S')
download_end_dt = datetime.strptime(end_time, '%d-%m-%Y %H:%M:%S')
req_df = bigtable_data_downloader(assetcode, customer, download_start_dt, download_end_dt)

def model_pred(req_df):
    with open('/home/srinivas/SONA_BLW/Ascent_descent_fp/ascent_descent_assembly.pkl', "rb") as pickle_file:
        ann_model = joblib.load(pickle_file)
    #import pickle
    #filename = '/home/srinivas/SONA_BLW/Ascent_descent_fp/finalized_model.sav'
    #ann_model = pickle.load(open(filename, 'rb'))
    x_test = req_df[['AB_BEARING_ASCENT_TEMP', 'AB_BEARING_DESCENT_TEMP',
                     'AB_BEARING_ASCENT_VIBRATION', 'AB_BEARING_DESCENT_VIBRATION']]
    y_preds = ann_model.predict(x_test)
    y_preds = [item for sublist in y_preds for item in sublist]
    preds = pd.DataFrame([y_preds]).T
    preds.columns = ['forecast']

    return preds

preds = model_pred(req_df)


def get_result(pred):
    temp1 = pred[pred['forecast'] == 1]
    if temp1.shape[0] > 0.6 * pred.shape[0]:
        result = 1
    else:
        result = 0

    return result


result = get_result(preds)
HydMotor_json={
            "customer":"SONA",
            "asset":"Rough Press",
            "asset code":"SONA01UNIT01ASST01RGH01",
            "component":"Ascent/Descent Assembly",
            "Date":datetime.today(),
            "Asset_Status": result
             }

db = firestore.client()
db.collection(u'Asset_OEE').document(u'SONA').collection(u'failure_pred1').document().set(HydMotor_json)
print('file uploaded')
