import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

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
    output='Nothing'
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if(prediction[2]==0):
        output='Polymer'
    elif(prediction[1]==1):
        output='Metal'
    elif(prediction[0]==1):
        output='Ceramic'

    return render_template('index.html', prediction_text='The material is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
