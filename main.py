import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection System")
st.sidebar.markdown(
    """
    <style>
    .stSidebar {
        background: linear-gradient(135deg, #56CCF2, #2F80ED);
        color: white;
        font-size: 1.2rem;
        padding: 10px;
    }
    .css-1d391kg {color: white;}
    </style>
    """,
    unsafe_allow_html=True
)
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Disease Detection"])

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
        color: #05445E;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #05445E;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        font-weight: bold;
    }
    .css-18e3th9 {
        padding: 1.5rem;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #189AB4;
        color: white;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #05445E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Page
if app_mode == "Home":
    st.title("🌱 Welcome to Plant Disease Detection System")
    st.image("home_page.jpeg", use_container_width=True)

    st.markdown(
        """
        ### Welcome to the Plant Disease Detection System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Detection** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        
        Designed by **Tazeem Hussain** and **Faizan Mirza**.
        """
    )

elif app_mode == "About":
    st.title("📖 About")
    st.markdown(
        """
        #### About Dataset
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The dataset is divided into an 80/20 ratio for training and validation while maintaining the directory structure.
        
        #### Content
        - **Train:** 70,295 images
        - **Test:** 33 images
        - **Validation:** 17,572 images

        The dataset was curated to aid in the detection of plant diseases and improve agricultural productivity.

        Designed by **Tazeem Hussain** and **Faizan Mirza**.
        """
    )

elif app_mode == "Disease Detection":
    st.title("🩺 Disease Detection")
    st.markdown("Upload an image of a plant leaf to detect potential diseases.")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Disease"):
            with st.spinner("Analyzing Image..."):
                result_index = model_prediction(test_image)

            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            st.success(f"The model predicts: **{class_name[result_index]}**")
            st.balloons()

# Footer
st.markdown(
    """
    <style>
    footer {
        visibility: visible;
        background-color: #05668D;
        color: white;
        padding: 10px;
        text-align: center;
        border-radius: 10px;
    }
    </style>
    <footer>
    Plant Disease Detection System | Powered by TensorFlow and Streamlit | Designed by Tazeem Hussain & Faizan Mirza
    </footer>
    """,
    unsafe_allow_html=True
)
