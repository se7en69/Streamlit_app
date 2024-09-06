import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Cache the model loading to ensure it's only done once
@st.cache_resource
def load_model_once():
    model = load_model('model4.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cache the encoders and scaler to ensure they're only loaded once
@st.cache_resource
def load_encoders_and_scaler():
    with open('Source_encoder.pkl', 'rb') as f:
        source_encoder = pickle.load(f)
    with open('Family_encoder.pkl', 'rb') as f:
        family_encoder = pickle.load(f)
    with open('Species_encoder.pkl', 'rb') as f:
        species_encoder = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return source_encoder, family_encoder, species_encoder, scaler, feature_names

# Load the model and encoders/scalers once
model = load_model_once()
source_encoder, family_encoder, species_encoder, scaler, feature_names = load_encoders_and_scaler()

# Cache data loading to prevent reloading
@st.cache_data
def load_data():
    try:
        data = pd.read_excel('MODEL_DATA.xlsx', sheet_name='Sheet1')
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

data = load_data()

if data is not None:
    # Extract unique options only from the model's training data
    species_options = sorted(species_encoder.classes_.tolist())
    family_options = sorted(family_encoder.classes_.tolist())
    source_options = sorted(source_encoder.classes_.tolist())

    # Dynamically extract antibiotic columns based on '_I' suffix in the column name
    antibiotic_columns = [col for col in data.columns if '_I' in col]

    # Define genotype columns
    genotype_columns = ['AMPC', 'SHV', 'TEM', 'CTXM1', 'CTXM2', 'CTXM825', 'CTXM9', 'VEB', 'PER', 'GES', 'ACC', 'CMY1MOX', 'CMY11', 'DHA', 'FOX', 'ACTMIR', 'KPC', 'OXA', 'NDM', 'IMP', 'VIM', 'SPM', 'GIM']

    def preprocess_input(form_data):
        try:
            # Convert form data to DataFrame
            input_data = pd.DataFrame([form_data])

            # Encode categorical features using previously fitted label encoders
            input_data['Source'] = source_encoder.transform(input_data['Source'])
            input_data['Family'] = family_encoder.transform(input_data['Family'])
            input_data['Species'] = species_encoder.transform(input_data['Species'])

            # Map antibiotic statuses
            for col in antibiotic_columns:
                input_data[col] = input_data[col].map({'Susceptible': 0, 'Intermediate': 1, 'Resistant': 2})

            # Ensure input data has all necessary columns in the correct order
            X = input_data[feature_names]

            # Scale the input data
            X = scaler.transform(X)
            return X
        except KeyError as e:
            st.error(f"Key error during preprocessing: {e}")
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 10px;
        }
        .stApp {
            background-color: #f0f2f6;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Streamlit app with custom title and description
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Gene Prediction from Antibiotic Resistance 🧬</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Select the options below to predict the presence of genes based on antibiotic resistance profiles.</p>", unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        species = st.selectbox('Select Species:', species_options)
    with col2:
        family = st.selectbox('Select Family:', family_options)
    with col3:
        source = st.selectbox('Select Source:', source_options)

    # Organize antibiotic inputs in two columns
    st.subheader('Select Antibiotic Resistance Status:')
    col4, col5 = st.columns(2)

    form_data = {
        'Species': species,
        'Family': family,
        'Source': source
    }

    for i, antibiotic in enumerate(antibiotic_columns):
        if i % 2 == 0:
            with col4:
                form_data[antibiotic] = st.radio(
                    f'{antibiotic}:',
                    ('Susceptible', 'Intermediate', 'Resistant')
                )
        else:
            with col5:
                form_data[antibiotic] = st.radio(
                    f'{antibiotic}:',
                    ('Susceptible', 'Intermediate', 'Resistant')
                )

    # Prediction button
    if st.button('Predict Genes'):
        with st.spinner('Predicting...'):
            try:
                X = preprocess_input(form_data)
                predictions = model.predict(X)

                # Prepare the output in terms of gene presence
                prediction_results = {gene: int(pred > 0.5) for gene, pred in zip(genotype_columns, predictions[0])}

                st.subheader('Prediction Results')
                st.write('Genes presence (1: Present, 0: Not Present)')
                st.table(pd.DataFrame(prediction_results.items(), columns=['Gene', 'Presence']))
            except KeyError as e:
                st.error(f"Error in input data: {e}. Please check your input options.")
            except Exception as e:
                st.error(f"Unexpected error during prediction: {e}")
else:
    st.error("Data failed to load. Please check the dataset file.")
