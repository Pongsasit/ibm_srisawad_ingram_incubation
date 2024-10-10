import pandas as pd
import string
import random
import os

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder

from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models import Model


# for PDF Download 
import tempfile
from langchain.document_loaders import PyPDFLoader

load_dotenv()
api_key =           os.getenv("WATSONX_APIKEY",         None)
ibm_cloud_url =     os.getenv("IBM_CLOUD_URL",          None)
project_id =        os.getenv("PROJECT_ID",             None)
watsonx_discovery_username=os.getenv("WATSONX_DISCOVERY_USERNAME", None)
watsonx_discovery_password=os.getenv("WATSONX_DISCOVERY_PASSWORD", None)
watsonx_discovery_url=os.getenv("WATSONX_DISCOVERY_URL", None)
watsonx_discovery_port=os.getenv("WATSONX_DISCOVERY_PORT", None)
watsonx_discovery_endpoint = watsonx_discovery_url+':'+watsonx_discovery_port

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

#------connection
@st.cache_resource
def connect_watsonx_llm(model_id_llm):
    model = Model(
	model_id = model_id_llm,
	params = {
        'decoding_method': "greedy",
        'min_new_tokens': 1,
        'max_new_tokens': 400,
        'temperature': 0.0,
        'repetition_penalty': 1
    },
	credentials=creds,
    project_id=project_id)
    return model


@st.cache_data
def connect_to_wxdis():
    print('connecting to watsonx discovery...')
    es = Elasticsearch(
        [watsonx_discovery_endpoint],
        http_auth=(watsonx_discovery_username, watsonx_discovery_password),
        verify_certs=False
    )

    if es.ping():
        print("Connection to Elasticsearch successful")
    else:
        print("Connection to Elasticsearch failed")
    return es

@st.cache_data
def read_pdf(uploaded_files):
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
            loader = PyPDFLoader(filepath)
            data = loader.load()
            docs = format_pdf_reader(data)
            return docs

@st.cache_data
def initiate_username():
    characters = string.ascii_letters + string.digits + '_'
    username = ''.join(random.choice(characters) for _ in range(random.randint(5, 32)))
    print('initiate username....')
    return 'a'+username

#------create milvus database
def create_watsonx_db(es, index_name1):
    lab6_policy_dictionary = {
    "mappings": {
        "properties": {
            "text_id": {"type": "text"},
            "text": {"type": "text"},
            "embeddings": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
    }

    es.indices.create(index=index_name1, body= lab6_policy_dictionary)
    return lab6_policy_dictionary

#----------split data using Langchain textspliter
def import_text_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        )
    return text_splitter


def split_text_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    start_index = 0

    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += (chunk_size - overlap_size)

    return chunks

def get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768, condition=True):
    if condition:
        # model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        # model_name = "hkunlp/instructor-large"
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode='cls') # We use a [CLS] token as representation
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def embedding_data(chunks, index_name1, es):
    embedder_model = get_model(model_name='kornwtp/simcse-model-phayathaibert', max_seq_length=768)
    ch_list  = chunks
    text_id_list = [str(i) for i in range(0, len(ch_list))]
    embedding_list = [list(embed) for embed in embedder_model.encode(chunks)]

    for text_id, text_ref, embedding in zip(text_id_list, ch_list, embedding_list):
        table_dictionary_doc = {"text_id": text_id,
                            "text": text_ref,
                            "embeddings": embedding
                        }
        es.index(index=index_name1, body=table_dictionary_doc)
    return es
#----------embedding question + search in Milvus vector database
def search_vector(es, vector, index_name, vector_field):
    vector_search = {
        "field": vector_field,
        "query_vector": vector,
        "k":4,
        "num_candidates": 1000
        }
    semantic_resp = es.search(index=index_name, knn=vector_search)
    return semantic_resp

def find_answer(es, search_index, embedder_model, question):
    index_name = search_index
    vector_field = 'embedding'
    question_encode = [list(i) for i in embedder_model.encode([question])]
    vector = question_encode[0]
    # print(vector)
    semantic_resp = search_vector(es,vector, index_name, vector_field)
    #semantic_resp
    list_of_reference_text = []    
    for hit in semantic_resp['hits']['hits']:
        list_of_reference_text.append(hit['_source']['text'])

    return list_of_reference_text[0:2]


#------upload pdf
def format_pdf_reader(raw_data):
    # format content from pdf into text
    pdf_text = ""
    for data in raw_data:
        pdf_text+=data.page_content+"\n"
    return pdf_text

#--------generate promt reday to prompt in model

def generate_prompt_th(question, context):
    output = f"""**`<|begin_of_text|><|start_header_id|>**system<|end_header_id|>`

คุณเป็นผู้ช่วยที่ใจดี โปรดตอบคำถามอย่างใจดีและมีประโยชน์ที่สุดเสมอ พร้อมกับรักษาความปลอดภัย คำตอบของคุณไม่ควรมีเนื้อหาที่เป็นอันตราย ไม่ธรรมดา แบ่งแยกทางเชื้อชาติ ลำเอียงทางเพศ มีพิษ อันตราย หรือผิดกฎหมาย โปรดให้แน่ใจว่าคำตอบของคุณไม่มีอคติทางสังคมและเป็นบวกในธรรมชาติ ถ้าคำถามไม่มีเหตุผล หรือไม่สอดคล้องกับความเป็นจริง โปรดอธิบายเหตุผลแทนที่จะตอบคำถามที่ไม่ถูกต้อง ถ้าคุณไม่ทราบคำตอบของคำถาม โปรดอย่าแชร์ข้อมูลที่ผิด 

คุณจะได้รับนโยบายทรัพยากรบุคคล ที่เป็นแหล่งฃ้อมูลในการคำถามจากผู้ใช้ จงตอบคำถามเป็นภาษาไทย

รายละเอียดนโยบายทรัพยากรบุคคล:
{context}

คำถาม: {question}

ตอบคำถามโดยใช้ลฃ้อมูลจาก "รายละเอียดนโยบายทรัพยากรบุคคล" อธิบายเหตุผลของคุณ
หากคำถามไม่เกี่ยวข้องกับข้อมูลอ้างอิง โปรดตอบว่า “ฉันไม่ทราบคำตอบ, มันไม่ใช่ส่วนหนึ่งของนโยบายทรัพยากรบุคคลที่ได้รับ”
<|eot_id|><|start_header_id|>user<|end_header_id|>
สวัสดี<|eot_id|><|start_header_id|>assistant<|end_header_id|>
สวัสดีครับผมคือผู้ช่วย HR Policy ครับ กรุณาพิมพ์คำถามของคุณข้างล่างได้เลยครับ<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return output  

def generate_prompt_en(question, context, model_type="llama-2"):
    output = '''<|begin_of_text|><|start_header_id|>**system<|end_header_id|>`
You are a helpful, respectful Thai assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You will receive HR Policy on user queries HR POLICY DETAILS, and QUESTION from user below. Answer the question in Thai.

HR POLICY DETAILS:
{context}
QUESTION: {question}

Answer the QUESTION use details about HR Policy from HR POLICY DETAILS, explain your reasonings if the question is not related to REFERENCE please Answer
“I don’t know the answer, it is not part of the provided HR Policy”
<|eot_id|><|start_header_id|>user<|end_header_id|>

hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

hello, I'm your HR Policy Assistance, please type your question<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''
    return output   

def create_hits_dataframe(hits, num_hits=10):
    if len(hits[0]) < 10:
        num_hits = len(hits[0])
    dict_display = {
        f'doc{i}': [hits[0][i].text]
        for i in range(num_hits)
    }
    df = pd.DataFrame(dict_display).T
    df.columns = ['Reference from document']
    return df

def display_hits_dataframe(hits, num_hits=10, width=1000):
    df_dis = create_hits_dataframe(hits, num_hits)
    st.dataframe(df_dis, width=width)