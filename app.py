import pickle
from flask import Flask , request, app,jsonify,url_for,render_template

import numpy as np
import pandas as pd


app = Flask(__name__)

models = pickle.load(open('classification.pkl','rb'))

scaler = pickle.load(open('scaling.pkl','rb'))

param = pickle.load(open('param.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=["POST"])
def predict_api():
    data = request.json
    print(data)

    new_data = np.array(list(data.values())).reshape(1,-1)

    column = ["age","sex","chest_pain_type","resting_blood_pressure","cholesterol","fasting_blood_sugar",	"resting_electrocardiogram","max_heart_rate_achieved","exercise_induced_angina","oldpeak","st_slope"]
    new_data = pd.DataFrame(new_data)

    new_data.columns = column

    new_data = scaler.transform(new_data)

    report = []

    for i in range(len(models)):

        model = list(models.values())[i]

        y_pred_test = model.predict(new_data)

        report.append(y_pred_test)

    positive = 0
    print(report)
    for i in range(len(report)):
        if report[i][0]==1:
            print(1)
            positive=positive+1
    
    if positive > 3:
        print("Heart Disease")
    else :
        print("Normal")
    
    print(positive)
    return jsonify(0)

@app.route("/predict",methods=["GET","POST"])
def predict():

    data = [float(x) for x in request.form.values()]
    print(data)
    new_data = np.array(data).reshape(1,-1)

    column = ["age","sex","chest_pain_type","resting_blood_pressure","cholesterol","fasting_blood_sugar",	"resting_electrocardiogram","max_heart_rate_achieved","exercise_induced_angina","oldpeak","st_slope"]
    new_data = pd.DataFrame(new_data)

    new_data.columns = column

    new_data = scaler.transform(new_data)

    report = []

    for i in range(len(models)):

        model = list(models.values())[i]

        y_pred_test = model.predict(new_data)

        report.append(y_pred_test)

    positive = 0
    print(report)
    for i in range(len(report)):
        if report[i][0]==1:
            print(1)
            positive=positive+1
    
    output = ''
    if positive > 3:
        output = 'Heart Disease'
    else :
        output = 'Normal'
    
    print(positive)
    return render_template("home.html",prediction_text = f'The Prediction is : {output}')

if __name__=="__main__":
    app.run(debug=True)