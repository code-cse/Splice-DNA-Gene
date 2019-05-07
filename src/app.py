from flask import Flask, request, jsonify, send_file
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin

import requests
import time 
import os
import base64
import predict as pred
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/seq', methods = ['POST'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def get_seq_data():
    
    if request.method == 'POST':
        print("in id input")
        data = request.stream.read()
        _list = []
        result = []
        for i in range(0,len(data), 60):
            data_list = []
            data_list.append(data[i:i+60])
            res = pred.predict_(data_list)

            result.append(res)
            _list.append(data[i:i+60])
        print(result)
        EI_Donor = result.count('Donor-EI')
        IE_Acceptor = result.count('Acceptor-IE')
        No_Junction = result.count('No-Junction')

        print("result", result)
        return jsonify({"result": result, "EI_Donor" : EI_Donor, \
            "IE_Acceptor" : IE_Acceptor, "No_Junction" : No_Junction})
    
    
if __name__ == '__main__':
   app.run(port=9900, host='0.0.0.0', debug=True)

