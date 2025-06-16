import streamlit as st
from streamlit_lottie import st_lottie
from huggingface_hub import InferenceClient
import requests
import json
import os

# --- 1. Initialize Session State ---
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "gender": None,
        "messages": [],
        "client": InferenceClient(
            model="deepseek-ai/DeepSeek-R1-0528",
            token=st.secrets["HF_TOKEN"]  # Using Streamlit secrets
        )
    }

# --- 2. Animation Loader ---
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Animation load error: {str(e)}")
        return None

# Load animations from local files
try:
    with open("Animation - 1749395326494.json", "r") as f:
        FEMALE_ANIM = json.load(f)

    with open("Animation - 1749394556693.json", "r") as f:
        CUSTOM_MALE = json.load(f)
except Exception as e:
    st.error(f"Failed to load animations: {str(e)}")
    st.stop()

# --- 3. Gender Selection UI ---
def show_gender_selection():
    st.markdown("""
    <style>
    .gender-title {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .gender-option {
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s;
        text-align: center;
    }
    .gender-option:hover {
        background: #f0f2f6;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="gender-title">🤖 Please select your gender :)</p>', unsafe_allow_html=True)

    male_anim = CUSTOM_MALE
    female_anim = FEMALE_ANIM

    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st_lottie(male_anim, height=200, key="male_anim")
            if st.button("Male", key="male_btn"):
                st.session_state.user_data["gender"] = "male"
                st.rerun()

    with col2:
        with st.container():
            st_lottie(female_anim, height=200, key="female_anim")
            if st.button("Female", key="female_btn"):
                st.session_state.user_data["gender"] = "female"
                st.rerun()

# --- 4. Chatbot Functionality with DeepSeek ---
def response_generator(prompt):
    try:
        # Prepare message history with system context
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant chatting with a {st.session_state.user_data['gender']} user."
            }
        ]
        
        # Add previous conversation history
        for msg in st.session_state.user_data["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add the new user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get streaming response
        completion = st.session_state.user_data["client"].chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
            stream=True
        )
        
        # Stream the response chunks
        full_response = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content
                full_response += chunk_content
                yield chunk_content
                
    except Exception as e:
        error_msg = f"🚫 Error: {str(e)}"
        st.error(error_msg)  # Show error in UI
        yield error_msg
        full_response = error_msg

# --- 5. Main App Flow ---
if not st.session_state.user_data["gender"]:
    show_gender_selection()
else:
    st.title(f"Yoo, {st.session_state.user_data['gender'].title()}😉")
    st.write("How can I help you today?")
    
    # Display chat history
    for message in st.session_state.user_data["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.user_data["messages"].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
            
        # Add assistant response to history
        st.session_state.user_data["messages"].append({"role": "assistant", "content": response})
        