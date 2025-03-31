import streamlit as st
import tensorflow as tf
import numpy as np
#tensor model prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_model.h5")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    return np.argmax(prediction)
#Sidebar
st.sidebar.title("DashBoard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

if(app_mode=="Home"):
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path="vegetable_fruit_image.jpg"
    st.image(image_path)
    
#About project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.code("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("Train: Contains 100 images per category.")
    st.text("Test: Contains 10 images per category.")
    st.text("Validation: Contains 10 images per category.")
    
#prediction page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image=st.file_uploader("Choose an Image")
    if(st.button("Show Image")):
        st.image(test_image)
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        #Reading labels
        with open("labels.txt") as f:
            content=f.readlines()
        label=[]
       
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting It is {}".format(label[result_index]))      