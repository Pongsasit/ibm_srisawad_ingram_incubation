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

from nzpyida import IdaDataBase, IdaDataFrame

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from config import tmg_abstract_index, tmg_sql_index

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

def retreive_abstract_to_data(es, tmg_abstract_index, embedder_model, question, reranker_model):
    index_name = tmg_abstract_index
    vector_field = 'embedding'
    question_encode = [list(i) for i in embedder_model.encode([question])]
    vector = question_encode[0]
    # print(vector)
    semantic_resp = search_vector(es,vector, index_name, vector_field)
    #semantic_resp
    list_of_abstract_question = []
    list_of_abstract_to_data_example = []
    list_of_table_description = []
    
    for hit in semantic_resp['hits']['hits']:
        list_of_abstract_to_data_example.append(hit['_source']['abstract_to_data_example'])
        list_of_table_description.append(hit['_source']['table_description'])
        list_of_abstract_question.append(hit['_source']['abstract_question'])

    question_list = []
    for abs_question in list_of_abstract_question:
        question_list.append([question, abs_question])
    
    th_scores = reranker_model.predict(question_list)
    abstract_to_data_example_prompt = list_of_abstract_to_data_example[th_scores.argmax()]
    table_description_prompt = list_of_table_description[th_scores.argmax()]
    return abstract_to_data_example_prompt, table_description_prompt

def retreive_dataq_to_sql(es, tmg_sql_index, embedder_model, chosen_question, reranker_model):
    index_name = tmg_sql_index
    vector_field = 'embedding'  
    query_encode = [list(i) for i in embedder_model.encode([chosen_question])]
    vector = query_encode[0]
    print(len(vector))
    # print(vector)
    semantic_resp = search_vector(es, vector, index_name, vector_field)
    #semantic_resp
    list_of_ask_question = []
    list_of_question_example = []
    list_of_table_description = []
    # print(semantic_resp)
    print(semantic_resp['hits']['hits'][0]['_source'].keys())
    for hit in semantic_resp['hits']['hits']:
        list_of_ask_question.append(hit['_source']['ask_question'])
        list_of_question_example.append(hit['_source']['question_example'])
        list_of_table_description.append(hit['_source']['table_description'])

    question_list = []
    for ask_question in list_of_ask_question:
        question_list.append([chosen_question, ask_question])
    
    th_scores = reranker_model.predict(question_list)        
    question_example_prompt = list_of_question_example[th_scores.argmax()]
    table_description_prompt = list_of_table_description[th_scores.argmax()]
    return question_example_prompt, table_description_prompt

#### WATSONX.AI ####
def send_to_watsonxai(model,
                    prompts
                    ):
    assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"
    output = []
    for prompt in prompts:
        o = model.generate_text(prompt)
        output.append(o)
    return output
    