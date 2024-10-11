import logging
import os

#for UI
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler
from PIL import Image
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder
# for function
from function import (connect_to_wxdis, 
                      connect_watsonx_llm, 
                      create_watsonx_db,
                      read_pdf, 
                      generate_prompt_th, 
                      split_text_with_overlap, 
                      embedding_data, 
                      find_answer, 
                      display_hits_dataframe,
                      initiate_username,
                      get_model)


#---------- settings ----------- #

# model_id_llm='meta-llama/llama-3-8b-instruct'
model_id_llm='meta-llama/llama-3-1-8b-instruct'
model_id_emb="kornwtp/simcse-model-phayathaibert"
embedder_model =  get_model(model_name= model_id_emb, max_seq_length=768)
index_name = 'hr_thai_policy'


# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")

es = connect_to_wxdis()
model_llm = connect_watsonx_llm(model_id_llm)

handler = StdOutCallbackHandler()
_ = create_watsonx_db(es, index_name)
# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
    ''')
    st.text_area('LLM used', f'{model_id_llm}', height=10)
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='Powered by watsonx.ai')    

#===========================================================================================


if uploaded_files := st.file_uploader("กรุณาเลีอกไฟล์ PDF", accept_multiple_files=True):
    print('======',index_name,'======')
    # try:
    if 1==1:
        thai_text = read_pdf(uploaded_files)
        chunks = split_text_with_overlap(thai_text, 1000, 300)
        print('----- create new collection')
        semantic_resp = embedding_data(chunks, index_name, es)
    # except:
        print('index already exist')
else:
    print('dropped collection')

if uploaded_files :
    print('ready for input...')
    if user_question := st.text_input(
        "ถามคำถามเกี่ยวกับเอกสารของคุณ:"
    ): 
        print('processing...')
        hits = find_answer(es, index_name, embedder_model, user_question)
        prompt = generate_prompt_th(user_question, str(hits))
        response = model_llm.generate_text(prompt)
        st.text_area(label="Model Response", value=response, height=300)
        display_hits_dataframe(hits)