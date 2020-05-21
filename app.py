from flask import Flask,request,jsonify

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from digit_recognizer import predict_digit,predict_digit_cnn


app = Flask(__name__)
app.secret_key = "xFs12Vsxwjkt2353974"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/',methods=['POST'])
def get_digit():
    req_data = request.json
    test_x = np.array(req_data['test_x'])
    result_1 = str(predict_digit(test_x))
    result_2 = str(predict_digit_cnn(test_x))
    
    return jsonify({'result_1':result_1,'result_2':result_2}),201
    
    

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0',port=9860)