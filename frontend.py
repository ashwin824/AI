import streamlit as st
import backend as demo

# import importlib
# importlib.reload(demo)

st.title("📌 Ashwin's Memory Chatbot")
st.caption("Powered by DeepSeek, LangChain (few-shot) and AWS Bedrock.")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.memory = demo.demo_memory()
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "**About this bot**\n\n"
        "Ashwin's Assistant")    

# st.write(f"session memory - Before : {st.session_state.memory}") 
# st.write(f"session chat history - Before : {st.session_state.chat_history}")

if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# st.write(f"session memory - After : {st.session_state.memory}") 
# st.write(f"session chat history - After : {st.session_state.chat_history}")    

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Ask me anything ...")   

if input_text:     
    with st.chat_message("user"):
        st.markdown(input_text)
        st.session_state.chat_history.append({"role":"user","text":input_text})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, update_memory = demo.demo_conversation(
                input_text=input_text,
                memory=st.session_state.memory
            )    
        st.markdown(response)

    st.session_state.memory = update_memory
    st.session_state.chat_history.append({"role":"assistant", "text":response})            