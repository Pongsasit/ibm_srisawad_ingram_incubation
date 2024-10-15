import os
import csv
import requests
from requests.auth import HTTPBasicAuth
import ast
from dotenv import load_dotenv
import traceback
import pandas as pd
import itertools

from flask import Flask, render_template, request, jsonify, Response

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder

#ibm ntz
from nzpyida import IdaDataBase, IdaDataFrame

#ibm watsonx
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

#wx.discovery
from function import search_vector, retreive_reference
#wx.ai
from function import send_to_watsonxai, generate_stream


#common libs
from function import get_model

from prompt import create_policy_question

app = Flask(__name__)
# Elasticsearch endpoint with port
load_dotenv()
project_id = os.environ["PROJECT_ID"]
ibm_cloud_url = os.environ["IBM_CLOUD_URL"]
api_key = os.environ["API_KEY"]
watsonx_discovery_username=os.environ["WATSONX_DISCOVERY_USERNAME"]
watsonx_discovery_password=os.environ["WATSONX_DISCOVERY_PASSWORD"]
watsonx_discovery_url=os.environ["WATSONX_DISCOVERY_URL"]
watsonx_discovery_port=os.environ["WATSONX_DISCOVERY_PORT"]
watsonx_discovery_endpoint = watsonx_discovery_url+':'+watsonx_discovery_port

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

if __name__ == '__main__':
    app.run(debug=False, port=8080, host="0.0.0.0")