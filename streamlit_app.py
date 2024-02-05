import streamlit as st

st.set_page_config(page_title="🤗💬 hate speech detection using L")

def predict(data):
    clf = joblib.load(r"C:\Users\Nava J Abburi\workspaces\AI apps\HateSpeechdetectionUsingLSTM.h5")
    return clf.predict(data)
with st.sidebar:
    st.title('🤗💬 Tweet hate speech detection using LSTM')


