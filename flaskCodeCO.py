# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:08:46 2021

@author: vb080076
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('LogisticRegression_CO.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HomePageCO.html')

@app.route('/result',methods=['POST'])
def result():

    float_features = [float(x) for x in request.form.values()]
    for i in range(7):
        if float_features[i] == 0:
            float_features[i] = 0
        else:
            float_features[i] = np.log(float_features[i])
    if float_features[9] == 0:
        float_features[9] = 0
    else:
        float_features[9] = np.log(float_features[9])
    float_features.append(float_features[0]*float_features[1])
    float_features.append(float_features[9]*float_features[1])
    float_features.append(float_features[0]*float_features[7])
    float_features.append(float_features[9]*float_features[8])
    final_features = [np.array(float_features)]
    prediction = model.predict_proba(final_features)[:,1]

    output = round(prediction[0], 2)
    if output >=0.67:
        output_text = ".    With threshold 0.67, the application should be auto approved."
    else:
        output_text = ".    With threshold 0.67, the application should be rejected and be sent to the credit analyst."
        
    return render_template('Result.html', prediction_text='Prob Prediction should be {} {}'.format(output,output_text))


if __name__ == "__main__":
    app.run(debug=True)
