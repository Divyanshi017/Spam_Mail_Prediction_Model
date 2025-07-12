import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Title
st.title("📧 Email Spam Classifier")
st.write("Predicts if an email is **spam (0)** or **ham (1)**")

# --- Model Training (Directly from your notebook) ---
# Load data (replace with your actual data loading logic)
@st.cache_data
def load_data():
    df = pd.read_csv("mail_data.csv")  # Ensure this file is in the same directory
    df = df.where((pd.notnull(df)), '')
    df.loc[df['Category'] == 'spam', 'Category'] = 0
    df.loc[df['Category'] == 'ham', 'Category'] = 1
    return df

df = load_data()
x = df['Message']
y = df['Category'].astype('int')

# Initialize and train the vectorizer and model (cached to avoid re-training)
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    x_features = vectorizer.fit_transform(x)
    model = LogisticRegression()
    model.fit(x_features, y)
    return vectorizer, model

vectorizer, model = train_model()

# --- Streamlit UI ---
user_input = st.text_area("Paste the email text here:")

if st.button("Predict"):
    # Transform input and predict
    input_features = vectorizer.transform([user_input])
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0]
    # Display result
    st.subheader("Result")
    if prediction == 0:
        st.error(f"🚨 **Spam**")
        st.image("https://i.pinimg.com/736x/3f/4f/1c/3f4f1c9496bda87f89e326d814c2ba93.jpg", width=400)
    else:
        st.success(f"✅ **Ham**")
        st.image("https://i.pinimg.com/736x/9f/fd/48/9ffd48eb7b7ee422e5bddc0cb2e0580b.jpg", width=400)