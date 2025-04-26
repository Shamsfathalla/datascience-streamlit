import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import joblib
import json
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

# Set page config (must be first Streamlit command)
st.set_page_config(page_title="U.S. Housing Market Analysis", layout="wide")

# Initialize session state to track dataset, model, and hierarchy loading
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'hierarchy_loaded' not in st.session_state:
    st.session_state.hierarchy_loaded = False

# Load the dataset from zip file on GitHub with caching
@st.cache_data
def load_data():
    zip_url = "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/b6e40641db4093903e513cb38d08532d551d23ef/datasets.zip"
    try:
        response = requests.get(zip_url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            csv_file_name = next((f for f in zip_ref.namelist() if "dataset_features_encoded_capped.csv" in f), None)
            if csv_file_name:
                with zip_ref.open(csv_file_name) as csv_file:
                    df = pd.read_csv(csv_file)
                    # Mapping for readability
                    area_type_map = {0: 'Rural', 1: 'Suburban', 2: 'Urban'}
                    city_type_labels = {
                        0: 'Town',
                        1: 'Small City',
                        2: 'Medium City',
                        3: 'Large City',
                        4: 'Metropolis'
                    }
                    df['area_type_label'] = df['area_type'].map(area_type_map)
                    df['city_type_label'] = df['city_type'].map(city_type_labels)
                    return df
            else:
                return None
    except Exception as e:
        return None

# Load the model from GitHub with caching
@st.cache_resource
def load_model():
    model_url = "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0525e619509d1de4140c82cb0145954ef7ace470/xgb_house_price_model.joblib"
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        model = joblib.load(io.BytesIO(response.content))
        return model
    except Exception as e:
        return None

# Load the region-state hierarchy JSON from GitHub with caching
@st.cache_data
def load_region_state_hierarchy():
    json_url = "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/a34f45f1bd9b857876cd31bdbee35468b9330b36/region_state_hierarchy.json"
    try:
        response = requests.get(json_url)
        response.raise_for_status()
        hierarchy = json.loads(response.content)
        return hierarchy
    except Exception as e:
        return None

# Initialize transformers for numerical features
@st.cache_resource
def initialize_transformers(df):
    power_transformers = {}
    scalers = {
        "acre_lot": MinMaxScaler(),
        "house_size": MinMaxScaler(),
        "property_size": MinMaxScaler(),
        "price": MinMaxScaler(),
        "bed_bath_ratio": MinMaxScaler()
    }
    df_transform = df.copy()
    skew_columns = ['acre_lot', 'house_size', 'property_size', 'bed_bath_ratio']
    for col in skew_columns:
        if col in df_transform.columns:
            df_transform[col] = np.log1p(df_transform[col])
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            df_transform[col] = pt.fit_transform(df_transform[[col]])
            power_transformers[col] = pt
    if 'price' in df_transform.columns:
        df_transform['price'] = np.log1p(df_transform['price'])
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df_transform['price'] = pt.fit_transform(df_transform[['price']])
        power_transformers['price'] = pt
    for col, scaler in scalers.items():
        if col in df_transform.columns:
            scaler.fit(df_transform[[col]])
    return power_transformers, scalers

# Load dataset, model, and region-state hierarchy
df = load_data()
model = load_model()
region_state_hierarchy = load_region_state_hierarchy()

# Check if dataset, model, and hierarchy loaded successfully
if df is not None:
    if not st.session_state.dataset_loaded:
        st.toast("Successfully loaded dataset from GitHub zip", icon="✅")
        st.session_state.dataset_loaded = True
else:
    st.error("Failed to load dataset from GitHub or CSV file not found in the zip archive")
    st.stop()

if model is not None:
    if not st.session_state.model_loaded:
        st.toast("Successfully loaded XGBoost model from GitHub", icon="✅")
        st.session_state.model_loaded = True
else:
    st.error("Failed to load the XGBoost model from GitHub")
    st.stop()

if region_state_hierarchy is not None:
    if not st.session_state.hierarchy_loaded:
        st.toast("Successfully loaded region-state hierarchy from GitHub", icon="✅")
        st.session_state.hierarchy_loaded = True
else:
    st.error("Failed to load the region-state hierarchy JSON from GitHub")
    st.stop()

# Initialize transformers
power_transformers, scalers = initialize_transformers(df)

# Define model features
model_features = [col for col in df.columns if col not in ['price', 'area_type_label', 'city_type_label']]

# Custom CSS for buttons and layout
st.markdown("""
    <style>
    .graph-button {
        background-color: rgb(38, 39, 48);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 16px;
        margin: 0 5px 5px 0;
        cursor: pointer;
        font-size: 14px;
        display: inline-block;
        text-align: center;
    }
    .graph-button:hover {
        background-color: rgb(50, 51, 60);
    }
    .stButton > button {
        width: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("U.S. Housing Market Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", 
                         ["Home", 
                          "Regional Price Differences", 
                          "Bedrooms/Bathrooms Impact", 
                          "House Size by City Type" , 
                          "Urban/Suburban/Rural Prices",
                          "House Price Predictor"])

# Home section
if section == "Home":
    st.markdown('<h4 style="margin-top:-20px;">by Shams Fathalla</h4>', unsafe_allow_html=True)  
    st.header("Welcome to the U.S. Housing Market Analysis Dashboard")
    st.write("""
    This interactive dashboard provides insights into various aspects of the U.S. housing market. 
    Use the navigation panel on the left to explore different sections:
    - **Regional Price Differences**: Compare property prices across U.S. regions
    - **Bedrooms/Bathrooms Impact**: See how these features affect home prices
    - **House Size by City Type**: Explore average house sizes across different city types
    - **Urban/Suburban/Rural Prices**: Compare prices across different area types
    - **House Price Predictor**: Predict house price based on property features and geographical data
    """)
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             use_container_width=True)

# Regional Price Differences section
elif section == "Regional Price Differences":
    st.header("1. How do property prices differ between the different U.S. regions?")
    
    # Define the graphs for this section
    graphs = {
        "Average Property Price by Region": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Property%20Price%20by%20Region.png",
        "Average Population in 2024 by Region": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Population%20in%202024%20by%20Region.png",
        "Average Density by Region": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Density%20by%20Region.png"
    }
    
    import streamlit as st

# Helper layouts

def three_buttons_layout(button_labels, keys_prefix):
    # 5 columns: spacer, btn1, btn2 (center), btn3, spacer
    cols = st.columns([1, 2, 2, 2, 1])
    btns = []
    btns.append(cols[1].button(button_labels[0], key=f"{keys_prefix}_btn1"))
    btns.append(cols[2].button(button_labels[1], key=f"{keys_prefix}_btn2"))
    btns.append(cols[3].button(button_labels[2], key=f"{keys_prefix}_btn3"))
    return btns

def four_buttons_layout(button_labels, keys_prefix):
    # 7 columns: spacer, btn1, btn2, btn3, btn4, spacer
    cols = st.columns([1, 2, 2, 2, 2, 2, 1])
    btns = []
    btns.append(cols[1].button(button_labels[0], key=f"{keys_prefix}_btn1"))
    btns.append(cols[2].button(button_labels[1], key=f"{keys_prefix}_btn2"))
    btns.append(cols[3].button(button_labels[2], key=f"{keys_prefix}_btn3"))
    btns.append(cols[4].button(button_labels[3], key=f"{keys_prefix}_btn4"))
    return btns

# Your main Streamlit app page handling

section = st.sidebar.selectbox(
    "Select Analysis Section",
    [
        "Regional Price Differences",
        "Bedrooms/Bathrooms Impact",
        "House Size by City Type",
        "Urban/Suburban/Rural Prices"
    ]
)

if section == "Regional Price Differences":
    st.header("1. How do property prices differ between the different U.S. regions?")
    
    graphs = {
        "Average Property Price by Region": "https://path_to_image_1.png",
        "Average Population in 2024 by Region": "https://path_to_image_2.png",
        "Average Density by Region": "https://path_to_image_3.png"
    }
    
    btns = three_buttons_layout(list(graphs.keys()), "rpd")
    
    if btns[0]:
        selected_graph = list(graphs.keys())[0]
    elif btns[1]:
        selected_graph = list(graphs.keys())[1]
    elif btns[2]:
        selected_graph = list(graphs.keys())[2]
    else:
        selected_graph = list(graphs.keys())[0]
    
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    
    # Add insights or explanations here
    st.write("Insights and explanation for:", selected_graph)

elif section == "Bedrooms/Bathrooms Impact":
    st.header("2. How does the number of bedrooms and bathrooms affect home prices?")
    
    graphs = {
        "Bedrooms vs Price": "https://path_to_image_4.png",
        "Bathrooms vs Price": "https://path_to_image_5.png",
        "Bed/Bath Ratio vs Price": "https://path_to_image_6.png"
    }
    
    btns = three_buttons_layout(list(graphs.keys()), "bbi")
    
    if btns[0]:
        selected_graph = list(graphs.keys())[0]
    elif btns[1]:
        selected_graph = list(graphs.keys())[1]
    elif btns[2]:
        selected_graph = list(graphs.keys())[2]
    else:
        selected_graph = list(graphs.keys())[0]
    
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    st.write("Insights and explanation for:", selected_graph)

elif section == "House Size by City Type":
    st.header("3. What is the average house size per city types in the U.S.?")
    
    graphs = {
        "House Size by City": "https://path_to_image_7.png",
        "Property Size by City": "https://path_to_image_8.png",
        "Acre Lot by City": "https://path_to_image_9.png"
    }
    
    btns = three_buttons_layout(list(graphs.keys()), "hsc")
    
    if btns[0]:
        selected_graph = list(graphs.keys())[0]
    elif btns[1]:
        selected_graph = list(graphs.keys())[1]
    elif btns[2]:
        selected_graph = list(graphs.keys())[2]
    else:
        selected_graph = list(graphs.keys())[0]
    
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    st.write("Insights and explanation for:", selected_graph)

elif section == "Urban/Suburban/Rural Prices":
    st.header("4. How do prices fluctuate between urban, suburban and rural cities?")
    
    graphs = {
        "Price by Area Type": "https://path_to_image_10.png",
        "Property Size by Area": "https://path_to_image_11.png",
        "Population by Area": "https://path_to_image_12.png",
        "Density by Area": "https://path_to_image_13.png"
    }
    
    btns = four_buttons_layout(list(graphs.keys()), "usar")
    
    if btns[0]:
        selected_graph = list(graphs.keys())[0]
    elif btns[1]:
        selected_graph = list(graphs.keys())[1]
    elif btns[2]:
        selected_graph = list(graphs.keys())[2]
    elif btns[3]:
        selected_graph = list(graphs.keys())[3]
    else:
        selected_graph = list(graphs.keys())[0]
    
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    st.write("Insights and explanation for:", selected_graph)

# House Price Predictor section
elif section == "House Price Predictor":
    st.header("5. Predict House Price")
    st.write("Enter the details below to predict the house price based on property size, bedrooms, bathrooms, region, city type, area type, and city.")
    
    # Map numerical codes to labels
    city_type_map = {
        "Town": 0, "Small City": 1, "Medium City": 2, "Large City": 3, "Metropolis": 4
    }
    area_type_map = {
        "Rural": 0, "Suburban": 1, "Urban": 2
    }
    
    st.subheader("Select Geographic Attributes")
    # Region dropdown
    regions = sorted(region_state_hierarchy.keys())
    selected_region = st.selectbox("Select Region", ["Select Region"] + regions, index=0)
    
    # State dropdown
    states = sorted(region_state_hierarchy[selected_region].keys()) if selected_region != "Select Region" else []
    selected_state = st.selectbox("Select State", ["Select State"] + states, index=0)
    
    # City Type dropdown
    city_types = sorted(region_state_hierarchy[selected_region][selected_state].keys()) \
        if selected_region != "Select Region" and selected_state != "Select State" else []
    selected_city_type_label = st.selectbox("Select City Type", ["Select City Type"] + city_types, index=0)
    
    # Area Type dropdown
    area_types = sorted(
        region_state_hierarchy[selected_region][selected_state][selected_city_type_label].keys()
    ) if (selected_region != "Select Region" and selected_state != "Select State" and selected_city_type_label != "Select City Type") else []
    selected_area_type_label = st.selectbox("Select Area Type", ["Select Area Type"] + area_types, index=0)

    selected_city_type = city_type_map.get(selected_city_type_label, 0)
    selected_area_type = area_type_map.get(selected_area_type_label.lower(), 0)
    
    # City dropdown
    cities = []
    if (selected_region != "Select Region" and selected_state != "Select State" and 
        selected_city_type_label != "Select City Type" and selected_area_type_label != "Select Area Type"):
        try:
            cities = region_state_hierarchy[selected_region][selected_state][selected_city_type_label][selected_area_type_label]
            if not isinstance(cities, list):
                st.error("City list is not valid.")
                cities = []
        except (KeyError, TypeError) as e:
            st.error(f"Error loading cities: {e}")
    cities = sorted(cities) if cities else ["No cities available"]
    selected_city = st.selectbox("Select City", ["Select City"] + cities, index=0)

    # Input property info
    st.subheader("Input House Details")
    col1, col2 = st.columns(2)
    with col1:
        property_size = st.number_input("Property Size (sq ft)", min_value=100.0, max_value=100000.0, value=100.0, step=100.0)
        bed = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=1, step=1)
    with col2:
        bath = st.number_input("Number of Bathrooms", min_value=1, max_value=20, value=1, step=1)
    
    if st.button("Predict House Price"):
        bed_bath_ratio = bed / bath if bath != 0 else 1.0
        input_data = {
            'property_size': property_size,
            'bed': bed,
            'bath': bath,
            'bed_bath_ratio': bed_bath_ratio,
            'city_type': selected_city_type,
            'area_type': selected_area_type,
            'region_Midwest': 1 if selected_region == 'Midwest' else 0,
            'region_Northeast': 1 if selected_region == 'Northeast' else 0,
            'region_South': 1 if selected_region == 'South' else 0,
            'region_West': 1 if selected_region == 'West' else 0,
        }
        input_df = pd.DataFrame([input_data])
        
        # Transform numerical features
        for col in ['property_size', 'bed_bath_ratio']:
            if col in input_df.columns and col in power_transformers:
                input_df[col] = np.log1p(input_df[col])
                input_df[col] = power_transformers[col].transform(input_df[[col]])
                input_df[col] = scalers[col].transform(input_df[[col]])
        
        # Add missing features with defaults
        for feature in model_features:
            if feature not in input_df.columns:
                if feature in df.columns:
                    median_value = df[feature].median()
                    if feature in power_transformers and feature not in ['bed', 'bath']:
                        median_df = pd.DataFrame({feature: [median_value]})
                        median_df[feature] = np.log1p(median_df[feature])
                        median_df[feature] = power_transformers[feature].transform(median_df[[feature]])[0][0]
                        if feature in scalers:
                            median_df[feature] = scalers[feature].transform(median_df[[feature]])[0][0]
                        input_df[feature] = median_df[feature]
                    else:
                        input_df[feature] = median_value
                else:
                    input_df[feature] = 0
        
        input_df = input_df[model_features]
        
        # Predict and inverse-transform the price
        try:
            prediction = model.predict(input_df)[0]
            if 'price' in power_transformers:
                unscaled_prediction = scalers['price'].inverse_transform([[prediction]])[0][0]
                untransformed_prediction = power_transformers['price'].inverse_transform([[unscaled_prediction]])[0][0]
                final_prediction = np.expm1(untransformed_prediction)
            else:
                final_prediction = prediction
            st.success(f"**Predicted House Price in {selected_city}:** ${final_prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
