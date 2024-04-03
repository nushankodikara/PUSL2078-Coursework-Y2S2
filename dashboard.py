from joblib import load
import numpy as np
import pandas as pd
import streamlit as st

# Sidebar for numerical inputs
cap_diameter = st.sidebar.number_input("Cap Diameter (cm)", min_value=0.0, format="%.2f")
stem_height = st.sidebar.number_input("Stem Height (cm)", min_value=0.0, format="%.2f")
stem_width = st.sidebar.number_input("Stem Width (mm)", min_value=0.0, format="%.2f")

# Dropdown for 'habitatXseason' with names
habitat_season_options = {
    'da': 'Woods - Autumn', 'ds': 'Woods - Spring', 'du': 'Woods - Summer', 'dw': 'Woods - Winter',
    'ga': 'Grasses - Autumn', 'gs': 'Grasses - Spring', 'gu': 'Grasses - Summer', 'gw': 'Grasses - Winter',
    'la': 'Leaves - Autumn', 'ls': 'Leaves - Spring', 'lu': 'Leaves - Summer', 'lw': 'Leaves - Winter',
    'ma': 'Meadows - Autumn', 'ms': 'Meadows - Spring', 'mu': 'Meadows - Summer', 'mw': 'Meadows - Winter',
    'pa': 'Paths - Autumn', 'ps': 'Paths - Spring', 'pu': 'Paths - Summer', 'pw': 'Paths - Winter',
    'ha': 'Heaths - Autumn', 'hs': 'Heaths - Spring', 'hu': 'Heaths - Summer', 'hw': 'Heaths - Winter',
    'ua': 'Urban - Autumn', 'us': 'Urban - Spring', 'uu': 'Urban - Summer', 'uw': 'Urban - Winter',
    'wa': 'Waste - Autumn', 'ws': 'Waste - Spring', 'wu': 'Waste - Summer', 'ww': 'Waste - Winter'
}
selected_habitat_season = st.sidebar.selectbox("Habitat X Season", options=list(habitat_season_options.keys()), format_func=lambda x: habitat_season_options[x])

# Dropdown for 'Cap Shape' with names
cap_shape_options = {
    'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f',
    'sunken': 's', 'spherical': 'p', 'others': 'o'
}
selected_cap_shape = st.sidebar.selectbox("Cap Shape", options=list(cap_shape_options.keys()), format_func=lambda x: x.capitalize())

# Dropdown for 'Cap Color' with names
cap_color_options = {
    'brown': 'n', 'buff': 'b', 'gray': 'g', 'green': 'r', 'pink': 'p',
    'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y', 'blue': 'l',
    'orange': 'o', 'black': 'k'
}
selected_cap_color = st.sidebar.selectbox("Cap Color", options=list(cap_color_options.keys()), format_func=lambda x: x.capitalize())

# Dropdown for 'Bruise or Bleed' with names
bruise_bleed_options = {'bruises-or-bleeding': 't', 'no': 'f'}
selected_bruise_bleed = st.sidebar.selectbox("Does Bruise or Bleed", options=list(bruise_bleed_options.keys()), format_func=lambda x: x.replace('-', ' ').capitalize())

# Dropdown for 'Gill Attachment' with names
gill_attachment_options = {
    'adnate': 'a', 'adnexed': 'x', 'decurrent': 'd', 'free': 'e',
    'sinuate': 's', 'pores': 'p', 'none': 'f', 'unknown': '?'
}
selected_gill_attachment = st.sidebar.selectbox("Gill Attachment", options=list(gill_attachment_options.keys()), format_func=lambda x: x.capitalize())

# Dropdown for 'Gill Color' with names reused from 'Cap Color'
selected_gill_color = st.sidebar.selectbox("Gill Color", options=list(cap_color_options.keys()), format_func=lambda x: x.capitalize())

# Dropdown for 'Stem Color' with names reused from 'Cap Color'
selected_stem_color = st.sidebar.selectbox("Stem Color", options=list(cap_color_options.keys()), format_func=lambda x: x.capitalize())

# Dropdown for 'Ring Type' with names
ring_type_options = {
    'cobwebby': 'c', 'evanescent': 'e', 'flaring': 'r', 'grooved': 'g',
    'large': 'l', 'pendant': 'p', 'sheathing': 's', 'zone': 'z',
    'scaly': 'y', 'movable': 'm', 'none': 'f', 'unknown': '?'
}
selected_ring_type = st.sidebar.selectbox("Ring Type", options=list(ring_type_options.keys()), format_func=lambda x: x.capitalize())

# Demonstration of Selected Options (Example)
st.write("# Mushroom Edibility Recognition")
st.write("## How to use this app")
st.write("* Enter the mushroom's characteristics in the sidebar. The model will predict if it's edible or poisonous with an accuracy rating of 99.8% after tuning the model.")
st.write("* Simply press the predict button after entering all the values.")
st.write("### Guidelines for selecting options")
st.image("Assets/mushroom-structure.jpeg", caption="Mushroom Guide", use_column_width=True)
st.write("#### 1. Cap Diameter")
st.image("Assets/cap-diameter.png", caption="Example measurement of Cap Diameter", use_column_width=True)
st.write("* Add X and Y then divide by 2 to get the diameter.")
st.write("#### 2. Stem Height")
st.image("Assets/stem-height.webp", caption="Example measurement of Stem Height", use_column_width=True)
st.write("* Measure the height of the stem from the base to the cap.")
st.write("#### 3. Stem Width")
st.image("Assets/stem-width.webp", caption="Example measurement of Stem Width", use_column_width=True)
st.write("* Measure the width of the stem at the base.")
st.write("#### 4. Habitat X Season")
st.write("* Select the habitat and season where the mushroom was found.")
st.write("#### 5. Cap Shape")
st.image("Assets/cap-shape.jpeg", caption="Example of Cap Shape", use_column_width=True)
st.write("* Select the shape of the cap.")
st.write("#### 6. Cap Color")
st.image("Assets/cap-color.jpeg", caption="Example of Cap Color", use_column_width=True)
st.write("* Mushrooms will come in different colors. Select the color of the cap.")
st.write("#### 7. Bruise or Bleed")
st.image("Assets/bruise-bleed.jpeg", caption="Example of Bruise or Bleed", use_column_width=True)
st.write("* Cut the mushroom and check if the mushroom bruises or bleeds.")
st.write("#### 8. Gill Attachment")
st.image("Assets/gill-attachment.webp", caption="Example of Gill Attachment", use_column_width=True)
st.write("* Check how the gills are attached to the stem.")
st.write("#### 9. Gill Color")
st.image("Assets/gill-color.jpeg", caption="Example of Gill Color", use_column_width=True)
st.write("* Select the color of the gills.")
st.write("#### 10. Stem Color")
st.image("Assets/stem-color.jpeg", caption="Example of Stem Color", use_column_width=True)
st.write("* Select the color of the stem.")
st.write("#### 11. Ring Type")
st.image("Assets/ring-type.jpeg", caption="Example of Ring Type", use_column_width=True)
st.write("* Select the type of ring on the stem.")
st.write("### Note")
st.write("The model is based on the Secondary Mushroom Dataset from UCI Machine Learning Repository. The model has 43,000 instances and the trained model got an accuracy of 99.8% and The model is a Random Forest Classifier.")



# Assuming the setup code is as provided earlier

# Initialize an object (dictionary) for model input
model_input = {
    "cap-diameter": cap_diameter,
    "stem-height": stem_height,
    "stem-width": stem_width,
}

# Update model_input with habitatXseason options, setting selected to True and others to False
for option in habitat_season_options.keys():
    model_input[option] = (option == selected_habitat_season)

# Cap Shape
for shape, abbreviation in cap_shape_options.items():
    model_input[f"cap_shape_{abbreviation}"] = (shape == selected_cap_shape)

# Cap Color
for color, abbreviation in cap_color_options.items():
    model_input[f"cap_color_{abbreviation}"] = (color == selected_cap_color)

# Bruise or Bleed
for option, value in bruise_bleed_options.items():
    model_input[f"bruise_or_bleed_{value}"] = (option == selected_bruise_bleed)

# Gill Attachment
for attach, abbreviation in gill_attachment_options.items():
    model_input[f"gill_attachment_{abbreviation}"] = (attach == selected_gill_attachment)

# Gill Color (reusing cap_color_options for names)
for color, abbreviation in cap_color_options.items():
    model_input[f"gill_color_{abbreviation}"] = (color == selected_gill_color)

# Stem Color (reusing cap_color_options for names)
for color, abbreviation in cap_color_options.items():
    model_input[f"stem_color_{abbreviation}"] = (color == selected_stem_color)

# Ring Type
for ring, abbreviation in ring_type_options.items():
    model_input[f"ring_type_{abbreviation}"] = (ring == selected_ring_type)

# Display the constructed model_input object for verification
# st.write("Model Input:")
# st.json(model_input)

# Assuming the previous setup and model_input construction

# Load models (Assuming the models are in the same directory as the script)
@st.cache_data()
def load_models():
    rf_model = load('Models/randomForest.pkl')
    pca_model = load('Models/pca_transformation.pkl')
    return rf_model, pca_model


rf_model, pca_model = load_models()

# Predict button in the sidebar
if st.sidebar.button('Predict', use_container_width=True):
    # Convert model_input to DataFrame for PCA transformation
    input_df = pd.DataFrame([model_input])

    # Ensure the order of columns matches the training data
    # Assuming you have a list of columns in the same order as during model training
    expected_columns = column_names = [
        "cap-diameter", "stem-height", "stem-width",
        "da", "ds", "du", "dw",
        "ga", "gs", "gu", "gw",
        "ha", "hs", "hu",
        "la", "ls", "lu", "lw",
        "ma", "ms", "mu", "mw",
        "pa", "pu",
        "ua", "us", "uu",
        "wa", "wu",
        "cap_shape_b", "cap_shape_c", "cap_shape_f", "cap_shape_o", "cap_shape_p", "cap_shape_s", "cap_shape_x",
        "cap_color_b", "cap_color_e", "cap_color_g", "cap_color_k", "cap_color_l", "cap_color_n", "cap_color_o", "cap_color_p", "cap_color_r", "cap_color_u", "cap_color_w", "cap_color_y",
        "bruise_or_bleed_t",
        "gill_attachment_a", "gill_attachment_d", "gill_attachment_e", "gill_attachment_p", "gill_attachment_s", "gill_attachment_x",
        "gill_color_b", "gill_color_e", "gill_color_g", "gill_color_k", "gill_color_n", "gill_color_o", "gill_color_p", "gill_color_r", "gill_color_u", "gill_color_w", "gill_color_y",
        "stem_color_b", "stem_color_e", "stem_color_g", "stem_color_k", "stem_color_l", "stem_color_n", "stem_color_o", "stem_color_p", "stem_color_r", "stem_color_u", "stem_color_w", "stem_color_y",
        "ring_type_e", "ring_type_f", "ring_type_g", "ring_type_l", "ring_type_p", "ring_type_r", "ring_type_z"
    ]
      # Replace with the list of columns in the correct order
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Convert DataFrame to numpy array
    input_array = input_df.to_numpy()

    # Apply PCA transformation
    transformed_input = pca_model.transform(input_array)

    # Predict with RandomForest model
    prediction = rf_model.predict(transformed_input)

    # Display the prediction
    prediction_label = 'Edible' if prediction[0] == 0 else 'Poisonous'  # Update based on your model's labeling

    # Create a colored box to show the prediction
    color = 'green' if prediction[0] == 0 else 'red'  # Update based on your model's labeling
    st.sidebar.markdown(f'Result: <span style="color: {color}; font-size: large;"><b>{prediction_label}</b></span>', unsafe_allow_html=True)

