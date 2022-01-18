# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:35:31 2021

@author: vb080076
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('LogisticRegression_IND.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HomePageIND.html')

@app.route('/result',methods=['POST'])
def result():

    float_features = [float(x) for x in request.form.values()]
    for i in range(5):
        if float_features[i] == 0:
            float_features[i] = 0
        else:
            float_features[i] = np.log(float_features[i])
    float_features.append(float_features[0]*float_features[1])
    float_features.append(float_features[0]*float_features[5])
    final_features = [np.array(float_features)]
    prediction = model.predict_proba(final_features)[:,1]

    output = round(prediction[0], 2)
    if output >=0.68:
        output_text = ".    With threshold 0.68, the application should be auto approved."
    else:
        output_text = ".    With threshold 0.68, the application should be rejected and be sent to the credit analyst."
        
    return render_template('Result.html', prediction_text='Prob Prediction should be {} {}'.format(output,output_text))


if __name__ == "__main__":
    app.run(debug=True)
