import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.title('Spam SMS Detection')
text=st.text_input('Enter the SMS you recieved', placeholder="E.g., You won $1000! Claim now!")
with open('spam_model.pkl', 'rb') as f:
    loaded_model=pickle.load(f)
with open('count_vectorizer.pkl', 'rb') as t:
    vectorizer=pickle.load(t)
labels=['NOT SPAM', 'SPAM']
if st.button('Check'):
    v=vectorizer.transform([text])
    preds=loaded_model.predict(v)
    prediction=labels[int(preds[0])]
    if prediction==labels[0]:
        st.success(f"Your SMS is **{prediction}**")
    else:
        st.error(f"Your SMS is **{prediction}**")