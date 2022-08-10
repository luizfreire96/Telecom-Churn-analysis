import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    X = pd.DataFrame({'PaymentMethod':request.form.get("Payment Method"), 'Contract':request.form.get("Contract"), 'StreamingMovies':request.form.get("Streaming Movies"), 'StreamingTV':request.form.get("StreamingTV"), 'OnlineSecurity':request.form.get("InternetSecurity"), 'tenure':request.form.get("Tenure")}, index=[0])
    
    Xcat = ['PaymentMethod', 'Contract', 'StreamingMovies', 'StreamingTV', 'OnlineSecurity']
    Xnum = ['tenure']
    
    prediction = model.predict(X)

    return render_template('index.html', prediction_text='{}'.format(str(prediction[0])))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    return jsonify(prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
