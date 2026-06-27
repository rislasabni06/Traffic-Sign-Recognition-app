import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "traffic_sign_model.h5"
)

classes = {
0:"Speed limit 5km/h",
1:"Speed limit 15km/h",
2:"Dont Go straight",
3:"Dont Go Left",
4:"Dont Go Left or Right",
5:"Dont Go Right",
6:"Dont overtake from Left",
7:"No Uturn",
8:"No Car",
9:"No horn",
10:"No entry",
11:"No stopping",
12:"Speed limit (30km/h)",
13:"Go straight or right",
14:"Go straight",
15:"Go left",
16:"Go Left or right",
17:"Go Right",
18:"keep Left",
19:"keep Right",
20:"Roundabout mandatory",
21:"watch out for cars",
22:"Horn",
23:"Speed limit (40km/h)",
24:"Speed limit (50km/h)",
25:"Speed limit (60km/h)",
26:"Speed limit (70km/h)",
27:"Speed limit (80km/h)",
28:"Dont Go straight or left",
29:"Yield"
}
st.title("Traffic Sign Recognition")

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=['png','jpg','jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image,width=250)
    img = image.resize((64,64))
    img = np.array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)*100

    st.success(f"Prediction : {classes[class_id]}")
    st.write(f"Confidence : {confidence:.2f}%")
