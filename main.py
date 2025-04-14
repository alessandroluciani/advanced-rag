
import streamlit as st

from services.embedder import Embedder
from services.ragger import Ragger

# from config.setup import config


rag = Ragger()


st.set_page_config(
    page_title="SherlockX",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.title("SherlockX")
st.divider()

with st.sidebar:
    st.title("SherlockX")
    st.divider()
    if st.button("Embed documents"):
        _ = Embedder().start_embedding_200()
        _ = Embedder().start_embedding_500()
        _ = Embedder().start_embedding_1000()

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter prompt here.."):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(Ragger().ollama_streaming(prompt))
        st.session_state["messages"].append({"role": "assistant", "content": message})


