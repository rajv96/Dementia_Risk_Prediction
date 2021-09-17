from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sklearn
import convert

import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

model = pickle.load(open('./model scripts/model.pkl', 'rb'))


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

    form_values = list(request.form.values())[: -1]
    converted = convert.convert(form_values)
    converted = model.predict([np.array(converted)])[0]
    return render_template('index.html', prediction_text = str(converted))


if __name__ == '__main__':
    app.run()

