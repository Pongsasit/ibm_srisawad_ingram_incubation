import os
import csv
import requests
from requests.auth import HTTPBasicAuth
import ast
from dotenv import load_dotenv
import traceback
import pandas as pd
import itertools
import json

from flask import Flask, render_template, request, jsonify, Response

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder


#ibm watsonx
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

#wx.discovery
from function import search_vector, retreive_reference
#wx.ai
from function import send_to_watsonxai, generate_stream, auto_ai_price_prediction, image_scoring_prompt, final_scoring_function


#common libs
from function import get_model

from prompt import create_policy_question

app = Flask(__name__)
# Elasticsearch endpoint with port
load_dotenv()
# Load environment variables
load_dotenv()
# Retrieve environment variables
watsonx_api_key = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
ibm_cloud_iam_url = os.getenv("IAM_IBM_CLOUD_URL", None)
chat_url = os.getenv("IBM_WATSONX_AI_INFERENCE_URL", None)
api_key = os.getenv("API_KEY", None)
# IBM Watson Discovery credentials
watsonx_discovery_username = os.getenv("WATSONX_DISCOVERY_USERNAME", None)
watsonx_discovery_password = os.getenv("WATSONX_DISCOVERY_PASSWORD", None)
watsonx_discovery_url = os.getenv("WATSONX_DISCOVERY_URL", None)
watsonx_discovery_port = os.getenv("WATSONX_DISCOVERY_PORT", None)
watsonx_discovery_endpoint = f"{watsonx_discovery_url}:{watsonx_discovery_port}"
conn_ibm_cloud_iam = http.client.HTTPSConnection(ibm_cloud_iam_url)
payload = "grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey&apikey="+watsonx_api_key
headers = { 'Content-Type': "application/x-www-form-urlencoded" }
conn_ibm_cloud_iam.request("POST", "/identity/token", payload, headers)
res = conn_ibm_cloud_iam.getresponse()
data = res.read()
decoded_json=json.loads(data.decode("utf-8"))
access_token=decoded_json["access_token"]

creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

LLM_model_id =  "meta-llama/llama-3-405b-instruct"
creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

loan_llm_model_params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.TEMPERATURE: 0.33,
    GenParams.TOP_P: 0.67,
    GenParams.TOP_K: 82,
    GenParams.REPETITION_PENALTY: 1.0,
    GenParams.STOP_SEQUENCES:['\n\n'],
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS:4000
}

loan_llm_model = Model(
    model_id=LLM_model_id,
    params=loan_llm_model_params,
    credentials=creds,
    project_id=project_id)


embedder_model = get_model(model_name='kornwtp/simcse-model-phayathaibert', max_seq_length=768)
reranker_model = CrossEncoder("Pongsasit/mod-th-cross-encoder-minilm")

mju_gate_index = 'lab08_index'


@app.route('/live')
def live():
    return "ok"

@app.route('/question_answer', methods=['POST'])
def question_guide_line():
    data = request.json
    question_to_be_guide = data["employee_question"]
    question = question_to_be_guide

    prompt_scope = create_policy_question(question)
    o = send_to_watsonxai(loan_llm_model, [prompt_scope])
    data_driven_question_list = str(o[0])
    
    print("response_g[0] START")
    print(o[0])
    print("response_g[0] END")
    
    real_list = data_driven_question_list
    print(real_list)

    results = {"result": real_list}
    return results


@app.route('/question_answer_streaming', methods=['POST'])
def question_answer_streaming():
    data = request.json
    question_to_be_guide = data["employee_question"]
    question = question_to_be_guide
    prompt_scope = create_policy_question(question)
    return Response(generate_stream(prompt_scope, loan_llm_model), content_type='text/event-stream')

@app.route('/price_prediction', methods=['POST'])
def price_prediction():
    data = request.json
    
    required_fields = ["Make", "Model", "Year", "Engine Fuel Type", "Engine HP", 
                       "Engine Cylinders", "Transmission Type", "Driven_Wheels", 
                       "Number of Doors", "Vehicle Size", "Vehicle Style", 
                       "highway MPG", "city mpg", "Popularity", "Years Of Manufacture", 
                       "Front View Image", "Rear View Image", "Right View Image", 
                       "Left View Image"]

    # Check for required fields
    for field in required_fields:
        if field not in data:
            return {'error': f'{field} is required'}, 400

    # Extract data
    make = data["Make"]
    model = data["Model"]
    year = data["Year"]
    engine_fuel_type = data["Engine Fuel Type"]
    engine_hp = data["Engine HP"]        
    engine_cylinder = data["Engine Cylinders"]
    transmission_type = data["Transmission Type"]
    driven_wheels = data["Driven_Wheels"]
    number_of_doors = data["Number of Doors"]
    vehicle_size = data["Vehicle Size"]
    vehicle_style = data["Vehicle Style"]
    highway_mpg = data["highway MPG"]
    city_mpg = data["city mpg"]
    popularity = data["Popularity"]
    age = data["Years Of Manufacture"]
    front_view_base64 = data["Front View Image"]
    rear_view_base64 = data["Rear View Image"]
    right_view_base64 = data["Right View Image"]
    left_view_base64 = data["Left View Image"]

    try:
        response_autoai, value_autoai = auto_ai_price_prediction(api_key, make, model, year, 
            engine_fuel_type, engine_hp, engine_cylinder, transmission_type, 
            driven_wheels, number_of_doors, vehicle_size, vehicle_style, 
            highway_mpg, city_mpg, popularity, age)

        rp_fr, front_result = image_scoring_prompt('front', front_view_base64, chat_url, project_id, access_token)
        rp_re, back_result = image_scoring_prompt('rear', rear_view_base64, chat_url, project_id, access_token)
        rp_ri, right_result = image_scoring_prompt('right', right_view_base64, chat_url, project_id, access_token)
        rp_le, left_result = image_scoring_prompt('left', left_view_base64, chat_url, project_id, access_token)

        estimate_thb = final_scoring_function(float(value_autoai), float(front_result), 
                                               float(back_result), float(left_result), 
                                               float(right_result))
        return {'price': float(estimate_thb)}
    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(debug=False, port=8080, host="0.0.0.0")