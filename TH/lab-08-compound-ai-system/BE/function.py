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
import time

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
    