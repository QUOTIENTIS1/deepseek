import streamlit as st
from streamlit_lottie import st_lottie
from huggingface_hub import InferenceClient
import json
import os

# ====================== #
#    APP CONFIGURATION   #
# ====================== #
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
ANIMATION_FILES = {
    "male": "Animation - 1749394556693.json",
    "female": "Animation - 1749395326494.json"
}

# ====================== #
#    SECURITY HANDLING   #
# ====================== #
def get_hf_client():
    """Initialize HF client with proper error handling"""
    try:
        if "HUGGINGFACE_API_TOKEN" not in st.secrets:
            st.error("❌ Missing API key in secrets.toml")
            st.stop()
            
        return InferenceClient(
            model=MODEL_NAME,
            token=st.secrets["HUGGINGFACE_API_TOKEN"]
        )
    except Exception as e:
        st.error(f"🔐 Authentication failed: {str(e)}")
        st.stop()

# ====================== #
#      ANIMATION LOAD    #
# ====================== #
@st.cache_data
def load_animation(path):
    """Cache animations for better performance"""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"🎬 Failed to load animation: {str(e)}")
        st.stop()

# ====================== #
#    CHATBOT ENGINE     #
# ====================== #
def generate_response(prompt):
    messages = [
        {"role": "system", "content": f"Assistant talking to {st.session_state.gender} user"}
    ] + st.session_state.messages + [
        {"role": "user", "content": prompt}
    ]
    
    try:
        stream = st.session_state.client.chat_completion(
            messages=messages,
            stream=True,
            max_tokens=1000
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_text = chunk.choices[0].delta.content
                full_response += chunk_text
                yield chunk_text
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# ====================== #
#       MAIN APP        #
# ====================== #
def main():
    # Initialize session state
    if "client" not in st.session_state:
        st.session_state.client = get_hf_client()
    if "gender" not in st.session_state:
        st.session_state.gender = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Gender selection screen
    if not st.session_state.gender:
        st.title("Welcome to DeepSeek Chat!")
        cols = st.columns(2)
        
        with cols[0]:
            st_lottie(load_animation(ANIMATION_FILES["male"]), height=200)
            if st.button("Male"):
                st.session_state.gender = "male"
                st.rerun()
                
        with cols[1]:
            st_lottie(load_animation(ANIMATION_FILES["female"]), height=200)
            if st.button("Female"):
                st.session_state.gender = "female"
                st.rerun()
    
    # Chat interface
    else:
        st.title(f"DeepSeek Chat ({st.session_state.gender.title()})")
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Handle new messages
        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                st.write_stream(generate_response(prompt))

if __name__ == "__main__":
    main()
