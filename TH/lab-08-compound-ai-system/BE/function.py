import os
import csv
import requests
from requests.auth import HTTPBasicAuth

import pandas as pd
import itertools

from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

import time
import ast
import tempfile  # for PDF Download
from langchain.document_loaders import PyPDFLoader
import base64
import http.client
import json


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


def get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768, condition=True):
    if condition:
        # model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        # model_name = "hkunlp/instructor-large"
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode='cls') # We use a [CLS] token as representation
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def search_vector(es, vector, index_name, vector_field):
    vector_search = {
        "field": vector_field,
        "query_vector": vector,
        "k":4,
        "num_candidates": 1000
        }
    semantic_resp = es.search(index=index_name, knn=vector_search)
    return semantic_resp

def retreive_reference(es, tmg_abstract_index, embedder_model, question, reranker_model):
    index_name = tmg_abstract_index
    vector_field = 'embedding'
    question_encode = [list(i) for i in embedder_model.encode([question])]
    vector = question_encode[0]
    # print(vector)
    semantic_resp = search_vector(es,vector, index_name, vector_field)
    #semantic_resp
    list_of_reference = []

    for hit in semantic_resp['hits']['hits']:
        list_of_reference.append(hit['_source']['reference_text'])

    question_list = []
    for refence in list_of_reference:
        question_list.append([question, refence])
    
    th_scores = reranker_model.predict(question_list)
    reference_for_prompt = list_of_reference[th_scores.argmax()]
    return reference_for_prompt

#### WATSONX.AI ####
def send_to_watsonxai_streaming(model_llm, 
                                prompt, 
                                #params, 
                                ):

    assert not any(map(lambda prompt: len(prompt) < 1, prompt)), "make sure none of the prompts in the inputs prompts are empty"
    # model_llm = Model(model_name,
    #                 params=params, credentials=creds,
    #                 project_id=project_id)
    return model_llm.generate_text_stream(prompt)

def fix_encoding(text):
    try:
        # Attempt to decode the text using 'ISO-8859-1' and then re-encode it in 'UTF-8'
        fixed_text = text.encode('ISO-8859-1').decode('UTF-8')
    except UnicodeDecodeError:
        # If there's a decoding error, return the original text
        fixed_text = text
    return fixed_text

def generate_stream(prompt, model_llm):
    start = time.time()
    model_stream = send_to_watsonxai_streaming(model_llm,
                                            prompt)
    print ("time to init streaming: ", time.time() - start)
    for response in model_stream:
        print(response)
        wordstream = str(response)
        if wordstream:
            wordstream = fix_encoding(wordstream)
            yield wordstream
    end = time.time()
    print("Time step 4: ", end - start)

def send_to_watsonxai(model,
                    prompts
                    ):
    assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"
    output = []
    for prompt in prompts:
        o = model.generate_text(prompt)
        output.append(o)
    return output
    

def final_scoring_function(price_predict, front_result, back_result, left_result, right_result):
    divided_price = (price_predict/4)
    front_price = divided_price*(front_result/100)
    back_price = divided_price*(back_result/100)
    left_price = divided_price*(left_result/100)
    right_price = divided_price*(right_result/100)
    sum_price = (front_price+back_price+left_price+right_price)*35
    if sum_price > 1500000:
        sum_price = 900000

    return sum_price

def image_scoring_prompt(side, pic_string64, chat_url, project_id, access_token):
    image_data = base64.b64decode(pic_string64)
    # Write the binary data to an image file
    with open('test.jpeg', "wb") as image_file:
        image_file.write(image_data)
    
    pic = open("test.jpeg","rb").read()
    pic_base64 = base64.b64encode(pic)
    pic_string = pic_base64.decode("utf-8")

    system_content = """You always answer the questions with json formatting using with 2 keys, score and reason. \n\nAny JSON tags must be wrapped in block quotes, for example ```{'score': '99', 'reason': 'all good'}```. 
    You will be penalized for not rendering code in block quotes.\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information."""
    user_message = f"""The position of the car is {side},
    1. Give a score of 1-100 for the condition the side of the car when 100 is perfect 
    2. Give the reason of the score."""
    format_text = """
    Answer in JSON with format ```{'score': integer, 'reason': str}``` always start with ```
    """
    # print(user_message)
    body = {
       "messages": [
        #   {
        #      "role": "system",
        #      "content": system_content
        #   },
          {
             "role": "user",
             "content": [
                {
                   "type": "text",
                   "text": system_content+'\n\n'+user_message+'\n'+format_text,
                },
                {
                   "type": "image_url",
                   "image_url": {
                      "url": f"data:image/jpeg;base64, {pic_string}"
                   }
                }
             ]
          }
       ],
       "project_id": project_id,
       "model_id": "meta-llama/llama-3-2-90b-vision-instruct",
       "decoding_method": "greedy",
       "repetition_penalty": 1.1,
       "max_tokens": 900
    }
    headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {access_token}"
    }
    response = requests.post(
        chat_url,
        headers=headers,
        json=body
    )
    
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    
    data = response.json()
    print('__'*20)
    print(data)
    print('__'*20)
    try:
        dictionary_data = ast.literal_eval(data['choices'][0]['message']['content'].split("```")[1])
    except:
        dictionary_data = {'score':80, 'reason':'good condition'}

    return  dictionary_data, float(int(dictionary_data['score']))

def auto_ai_price_prediction(api_key, make, model, year, engine_fuel_type, engine_hp, engine_cylinder,
                            transmission_type, driven_wheels, number_of_doors, vehicle_size,
                            vehicle_style, highway_mpg, city_mpg, popularity, age):
    API_KEY = api_key
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    
    payload_scoring = {"input_data": [{"fields": [
                                "Make",
                                "Model",
                                "Year",
                                "Engine Fuel Type",
                                "Engine HP",
                                "Engine Cylinders",
                                "Transmission Type",
                                "Driven_Wheels",
                                "Number of Doors",
                                "Vehicle Size",
                                "Vehicle Style",
                                "highway MPG",
                                "city mpg",
                                "Popularity",
                                "Years Of Manufacture"
                        ], "values": [[
                                str(make),
                                str(model),
                                str(year),
                                str(engine_fuel_type),
                                str(engine_hp),
                                str(engine_cylinder),
                                str(transmission_type),
                                str(driven_wheels),
                                str(number_of_doors),
                                str(vehicle_size),
                                str(vehicle_style),
                                str(highway_mpg),
                                str(city_mpg),
                                str(popularity),
                                str(age)
                        ]]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/car_price_prediction/predictions?version=2021-05-01', 
                                     json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    data = response_scoring.json()
    value = data['predictions'][0]['values'][0][0]
    return data, value
    
