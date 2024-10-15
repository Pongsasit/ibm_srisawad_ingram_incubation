import streamlit as st
import requests
from dotenv import load_dotenv
import os
import time

BE_ENDPOINT = 'http://127.0.0.1:8080' + '/question_answer_streaming'

def get_response_streaming(data, endpoint='http://127.0.0.1:8001/question_answer_streaming'):
    payload = {'employee_question': data}
    r = requests.post(endpoint, json=payload, stream=True)
    return r

st.title("SRISAWAD Assistant Bot")

# Toggle for enabling/disabling chat history
if "record_chat_history" not in st.session_state:
    st.session_state.record_chat_history = False  # Default is True

st.session_state.record_chat_history = False

# Initialize the session state for storing messages if it does not exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add feedback buttons if message is from assistant and feedback hasn't been given yet
        if message["role"] == "assistant" and "feedback" not in message:
            like_button = st.button("Like", key=f"like_{i}")
            dislike_button = st.button("Dislike", key=f"dislike_{i}")

            if like_button:
                message["feedback"] = "save"
                st.session_state.messages[i] = message
                st.success("Feedback saved: 👍 Like")
                # st.experimental_rerun()
            if dislike_button:
                message["feedback"] = "unsave"
                st.session_state.messages[i] = message
                st.error("Feedback saved: 👎 Dislike")
                # st.experimental_rerun()

# User input
question = st.chat_input("มีอะไรให้ช่วยไหมคะ?")
if question:
    # Add user message to chat history and display immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        sources_place_holder = st.empty()
    print(f"FE init history")
    # Write chat history to file
    start = time.time()

    with st.spinner('loading'):
        r = get_response_streaming(question, BE_ENDPOINT)
        end = time.time()
        print(f"FE after r time: {end-start}")
        show = True
        # Display assistant response
        with st.chat_message("assistant"):
            model_response_placeholder = st.empty()
            full_response = ""

            print(f"FE raw text {r}")
            for chunk in r.iter_content(chunk_size=1024):
                print(chunk)
                if chunk:
                    response = chunk.decode('utf-8')
                    partial_message = response
                    full_response += partial_message  # Append the new chunk to the full message
                    with model_response_placeholder.container():
                        st.markdown(full_response)  # Display the updated full message
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Add feedback buttons after the assistant's response
        like_button = st.button("ประเมินคำตอบ", key=f"like_{len(st.session_state.messages) - 1}")
        # dislike_button = st.button("ข้ามการประเมินคำตอบ", key=f"dislike_{len(st.session_state.messages) - 1}")

        if like_button:
            # st.session_state.messages[-1]["feedback"] = "like"
            st.success("Feedback saved: 👍 Like")
