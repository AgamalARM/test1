from fastapi import FastAPI
from fastapi.params import Body
import requests
import pickle
import numpy as np
import pandas as pd
################################################
app = FastAPI()
#### to run the code >>>   'https://dexcom-ml.herokuapp.com/api/Dexcom_classification' ###
with open("finalized_model.sav", "rb") as f:
    trained_pickled_clf = pickle.load(f)
#############  Operation Functions          ##############
############   Get mapped of trend string   ##############


def get_mapped_data(trend):
    if trend == 'Flat':
        return 0.1
    if trend == 'Double up':
        return 0.2
    if trend == 'Double down':
        return 0.3
    if trend == 'Single up':
        return 0.4
    if trend == 'Single down':
        return 0.5
    if trend == 'Forty_five up':
        return 0.6
    if trend == 'Forty_five down':
        return 0.7

#########################################################
#######   Get result of Machine learning algorithm   ####


def get_result(student_id, reading, trend):
    mapped_trend = get_mapped_data(trend)
    input = [reading, mapped_trend]
    arr_input = np.asarray(input)
    reshapedArray = arr_input.reshape(1, -1)
    final_output_ML = trained_pickled_clf.predict(reshapedArray)
    classification_output = int(final_output_ML)
##### convert classification to json data to send to BackEnd ############
    if (classification_output == 3):
        return {"student_id": student_id, "value": reading, "trend": trend, "classification": 3, 'alert': "non"}
    elif (classification_output == 2):
        return {"student_id": student_id, "value": reading, "trend": trend, "classification": 2, 'alert': "yellow"}

    elif (classification_output == 1):
        return {"student_id": student_id, "value": reading, "trend": trend, "classification": 1, 'alert': "red"}

    elif (classification_output == 5):
        return {"student_id": student_id, "value": reading, "trend": trend, "classification": 5, 'alert': "red"}

    elif (classification_output == 4):
        return {"student_id": student_id, "value": reading, "trend": trend, "classification": 4, 'alert': "yellow"}

###########################################################
##########    API to get data from Simulation BackEnd   ########


@app.get("/api/Dexcom_classification")
async def root():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    x = requests.get('http://dexcom.invasso.com/api/dexcom/simulation', headers=headers)
    y = x.json()

    trend_name = y['trend']
    reading_value = y['sensor_treading_value']
    student_id = y['student_id']

    # Special Cases ###Rule==> there are {range & trend Flat & value within range}
    if (('range' in y) and (reading_value >= int(y['range']['from'])) and (reading_value <= int(y['range']['to'])) and (trend_name == 'Flat')):
        studentRange = y['range']
        return {"student_id": student_id, "Student_Range": studentRange, "value": reading_value, "trend": trend_name, "classification": 3, 'alert': "non"}

    else:

        return get_result(student_id, reading_value, trend_name)

    
