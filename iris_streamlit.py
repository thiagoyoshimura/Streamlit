import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

st.title("Iris Flower Classification")

st.sidebar.header("User Input")

sepal_length = st.sidebar.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.sidebar.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

user_input = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

prediction = clf.predict(user_input)

st.subheader("User Input:")
st.write(user_input)

st.subheader("Prediction:")
st.write(f"Iris species: {iris.target_names[prediction[0]]}")

