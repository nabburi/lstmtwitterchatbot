import streamlit as st
import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="🤗💬 hate speech detection using L")

def create_sidebar():
    st.sidebar.subheader("Please Select Your Use Case Here:")

    use_case = st.sidebar.radio(
        label="List of Use Cases:", 
        options=("Human or Machine Generated Speech", "Option 1", "Option 2", "Option 3"),
        index=0
        )
    return use_case

def remove_punctuations(text):
    punctuation_list = string.punctuation
    temp = str.maketrans('', '', punctuation_list)
    return text.translate(temp)

def remove_stopwords(text):
    imp_words = []
    #sort imp words
    for word in str(text).split():
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize(word)
        imp_words.append(word)
    output = " ".join(imp_words)
    return output

def main():
    use_Case = create_sidebar()
    st.subheader(use_Case)
    uploaded_file = ""
    input_text = ""
    te_dataset = pd.DataFrame()
    if(use_Case == "Human or Machine Generated Speech"):
        st.write("Given a text, the model will classify wheather the text is Hate Speech or Offensive Language or neither")
        if(use_Case == "Human or Machine Generated Speech"):
            input_file = st.radio(
                label="Select the type of input",
                options=("Enter text", "Sample input file", "upload the csv file"),
                index=1
            )
            if(input_file == "Enter text"):
                input_text = st.text_input("Please enter your text", "", key="placeholder")
            elif(input_file == "Upload the csv file"):
                uploaded_file = st.file_uploader("Please upload your csv file", type=["csv"])
            input_file_name = 'data/input/testdataset_hatesppech.csv'
            results_file_path = "data/results/testdatatset_hatespeech/"
            label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
            

        if st.button("Generate Results"):
            if(input_text != ""):
                te_dataset = pd.DataFrame(columns = ['text'])
                te_dataset['text'] = [input_text]
            elif(uploaded_file != ""):
                te_dataset = pd.read_csv(uploaded_file, sep="\t")
            else:
                te_dataset = pd.read_csv(input_file_name, sep="\t")
            #te_dataset["tokenized"] = te_dataset.apply(lambda row: len(nltk.word), axis=1))
        
            #text-preprocessing
            te_dataset.iloc[:, 0] = te_dataset.iloc[:, 0].str.lower()
            te_dataset.iloc[:, 0] = te_dataset.iloc[:, 0].apply(lambda x: remove_punctuations(x))
            te_dataset.iloc[:, 0] = te_dataset.iloc[:, 0].apply(lambda text: remove_stopwords(text))

            #tokenizer
            max_words = 5000
            token = Tokenizer(num_words=max_words,
                    lower=True,
                    split=' ')
            token.fit_on_texts(te_dataset.iloc[:, 0])
            Testing_seq = token.texts_to_sequences(te_dataset.iloc[:, 0])
            Testing_pad = pad_sequences(Testing_seq,
                                maxlen=100,
                                padding='post',
                                truncating='post')
            model = load_model("HateSpeechdetectionUsingLSTM.h5")
            evaluation = model.predict(Testing_pad)
            

            html_str = f""" 
                <div>&nbsp;</div>
                <div style="padding: 0px; width: 95%; align:center; border-top: 3px solid darkgray;">&nbsp;</div>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            pred = np.argmax(evaluation, axis = 1)
            print(pred)
            te_dataset['predicted_label'] = pred
            te_dataset['predicted_label'] = te_dataset['predicted_label'].apply(lambda x: label_map[x])
            st.write(te_dataset)


if __name__ == "__main__":
    main()
