#!/usr/bin/env python

from flask import Flask
from flask import abort
from flask import request, jsonify

from src import caffeprediction as caffeprediction
from src import tensorflowprediction as tensorflowprediction
from src.modeldetails import ModelDetails
import json
import os
import logging
import datetime
import werkzeug
import sys
import urllib2

app = Flask(__name__)

@app.route('/api/ping')
def ping():
	return 'pong'

@app.route('/api/predict')
def predict():
	model = request.args.get('model')
	filename = request.args.get('path')
	
	if model == None:
		return abort(400, {'message': 'Missing argument: model'})
	if filename == None:
		return abort(400, {'message': 'Missing argument: path'})

	details = fetch_model_details(model)
	details.validate()
	
	if details.provider == 'caffe':
		result = caffeprediction.predict(modeldetails, filename)
	elif details.provider == 'tensorflow':
		result = tensorflowprediction.predict(modeldetails, filename)
	else:
		return abort(400, {'message': 'Missing argument: path'})

	return jsonify(result)

def fetch_model_details(modelid):
	response = urllib2.urlopen("http://localhost:9080/api/modeldetails/" + modelid)
	jsonstr = response.read()
	decoded = json.loads(jsonstr)
	return ModelDetails(
		decoded[u'id'],
		decoded[u'userId'],
		decoded[u'name'],
		decoded[u'provider'],
		decoded[u'path']
	)

if __name__ == "__main__":
	# content = fetch_model_details("caffe")
	modeldetails = fetch_model_details("f92f738e-5f6c-46e1-8e3b-2eddaaafbc7a")
	print(modeldetails.provider)
	app.run(debug=True, host='0.0.0.0', threaded=False)
