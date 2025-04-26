import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import joblib
import json
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import plotly.graph_objects as go

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
    
    # Create centered container for 3 buttons
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        btn1 = st.button(list(graphs.keys())[0], key="btn1")
    with col2:
        btn2 = st.button(list(graphs.keys())[1], key="btn2")
    with col3:
        btn3 = st.button(list(graphs.keys())[2], key="btn3")
    
    # Determine which graph to show
    if btn1:
        selected_graph = list(graphs.keys())[0]
    elif btn2:
        selected_graph = list(graphs.keys())[1]
    elif btn3:
        selected_graph = list(graphs.keys())[2]
    else:
        selected_graph = list(graphs.keys())[0]  # Default
    
    # Display the selected graph
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    
    # Insights section
    st.write("""
    ### Key Insights:
    - West Region:
        - Property Price: The West continues to have the highest average property price at approximately $473,869.94.
        - Population and Density: Despite a moderate population density, the high property prices in the West suggest that factors beyond density contribute significantly to elevated housing costs.
    - Northeast Region:
        - Property Price: The Northeast has a high average property price of around $277,303.81.
        - Population and Density: The Northeast's high density (5,942.71) reflects its concentration of major cities.
    - South Region:
        - Property Price: The South has an average property price of about $296,500.84.
        - Population and Density: With a lower population density (3,302.13) than the Northeast and West.
    - Midwest Region:
        - Property Price: The Midwest has the lowest average property price at approximately $231,103.74.
        - Population and Density: The Midwest's low population density impacts housing demand.
    ### Final Answer:
    - Property Prices: West (Highest) > South > Northeast > Midwest (Lowest).
    - Driven by population density, urban development, economic activity, and regional demand.
    """)

# Bedrooms/Bathrooms Impact section
elif section == "Bedrooms/Bathrooms Impact":
    st.header("2. How does the number of bedrooms and bathrooms affect home prices?")
    
    # Define the graphs for this section
    graphs = {
        "Bedrooms vs Price": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Bedrooms%20vs%20Price.png",
        "Bathrooms vs Price": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Bathrooms%20vs%20Price.png",
        "Bed/Bath Ratio vs Price": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Bed&Bath%20Ratio%20vs%20Price.png"
    }
    
    # Create centered container for 3 buttons
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        btn1 = st.button(list(graphs.keys())[0], key="btn1")
    with col2:
        btn2 = st.button(list(graphs.keys())[1], key="btn2")
    with col3:
        btn3 = st.button(list(graphs.keys())[2], key="btn3")
    
    # Determine which graph to show
    if btn1:
        selected_graph = list(graphs.keys())[0]
    elif btn2:
        selected_graph = list(graphs.keys())[1]
    elif btn3:
        selected_graph = list(graphs.keys())[2]
    else:
        selected_graph = list(graphs.keys())[0]  # Default
    
    # Display the selected graph
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    
    # Insights section
    st.write("""
    ### Key Insights:
    - Number of Bedrooms: More bedrooms increase price, but marginal gains diminish.
    - Number of Bathrooms: More bathrooms increase price, with diminishing returns.
    - Bed/Bath Ratio: Optimal ratio (~1.25) maximizes price; imbalanced ratios lower prices.
    ### Final Answer:
    - Bedrooms and bathrooms positively affect prices, with diminishing returns. Optimal bed/bath ratio (~1.25) maximizes value.
    """)

# House Size by City Type section
elif section == "House Size by City Type":
    st.header("3. What is the average house size per city types in the U.S.?")
    
    # Define the graphs for this section
    graphs = {
        "House Size by City": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20House%20Size%20by%20City%20Type.png",
        "Property Size by City": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Property%20Size%20by%20City%20Type.png",
        "Acre Lot by City": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Acre%20Lot%20by%20City%20Type.png"
    }
    
    # Create centered container for 3 buttons
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        btn1 = st.button(list(graphs.keys())[0], key="btn1")
    with col2:
        btn2 = st.button(list(graphs.keys())[1], key="btn2")
    with col3:
        btn3 = st.button(list(graphs.keys())[2], key="btn3")
    
    # Determine which graph to show
    if btn1:
        selected_graph = list(graphs.keys())[0]
    elif btn2:
        selected_graph = list(graphs.keys())[1]
    elif btn3:
        selected_graph = list(graphs.keys())[2]
    else:
        selected_graph = list(graphs.keys())[0]  # Default
    
    # Display the selected graph
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    
    # Insights section
    st.write("""
    ### Key Insights:
    - House Sizes: Larger in towns/small cities (less dense); smaller in metropolises (dense).
    - Property/Lot Sizes: Larger in rural/suburban; smaller in urban.
    ### Conclusion
    - Average house sizes:
        - Town: 1,721.2669 sq ft
        - Small City: 1,753.2493 sq ft
        - Medium City: 1,692.1574 sq ft
        - Large City: 1,643.5799 sq ft
        - Metropolis: 1,621.2234 sq ft
    ### Final Answer:
    - House sizes decrease with increasing population density.
    """)

# Urban/Suburban/Rural Prices section
elif section == "Urban/Suburban/Rural Prices":
    st.header("4. How do prices fluctuate between urban, suburban and rural cities?")
    
    # Define the graphs for this section
    graphs = {
        "Price by Area Type": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Property%20Price%20by%20Area%20Type.png",
        "Property Size by Area": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Property%20Size%20by%20Area%20Type.png",
        "Population by Area": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Population%20(2024)%20by%20Area%20Type.png",
        "Density by Area": "https://raw.githubusercontent.com/Shamsfathalla/datascience-streamlit/0d4ccb38eae49fa972b94d44116c05c44b640f16/Images/Average%20Population%20Density%20by%20Area%20Type.png"
    }
    
    # Create centered container for 4 buttons
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        btn1 = st.button(list(graphs.keys())[0], key="btn1")
    with col2:
        btn2 = st.button(list(graphs.keys())[1], key="btn2")
    with col3:
        btn3 = st.button(list(graphs.keys())[2], key="btn3")
    with col4:
        btn4 = st.button(list(graphs.keys())[3], key="btn4")
    
    # Determine which graph to show
    if btn1:
        selected_graph = list(graphs.keys())[0]
    elif btn2:
        selected_graph = list(graphs.keys())[1]
    elif btn3:
        selected_graph = list(graphs.keys())[2]
    elif btn4:
        selected_graph = list(graphs.keys())[3]
    else:
        selected_graph = list(graphs.keys())[0]  # Default
    
    # Display the selected graph
    st.subheader(selected_graph)
    st.image(graphs[selected_graph], use_container_width=True)
    
    # Insights section
    st.write("""
    ### Key Insights:
    - Urban: Highest prices due to high density, limited land, and demand.
    - Suburban: Mid-range prices, balancing affordability and features.
    - Rural: Lowest prices, larger lots but less demand/amenities.
    ### Final Answer:
    - Prices increase from rural to suburban to urban due to density, land availability, and amenities.
    """)

# State name to abbreviation mapping
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Define your regions
northeast_states = {
    "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
    "New Jersey", "New York", "Pennsylvania"
}

midwest_states = {
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri", "Nebraska",
    "North Dakota", "Ohio", "South Dakota", "Wisconsin"
}

south_states = {
    "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky", "Louisiana", "Maryland",
    "Mississippi", "North Carolina", "Oklahoma", "South Carolina", "Tennessee", "Texas",
    "Virginia", "West Virginia"
}

west_states = {
    "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada",
    "New Mexico", "Oregon", "Utah", "Washington", "Wyoming"
}

def get_region_for_state(state):
    if state in northeast_states:
        return 'Northeast'
    elif state in midwest_states:
        return 'Midwest'
    elif state in south_states:
        return 'South'
    elif state in west_states:
        return 'West'
    else:
        return 'Other'

def plot_region_map(selected_state_name):
    # State abbreviation dictionary
    state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    # Define sets of states for regions
    northeast_states = {
        "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
        "New Jersey", "New York", "Pennsylvania"
    }
    midwest_states = {
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri", "Nebraska",
        "North Dakota", "Ohio", "South Dakota", "Wisconsin"
    }
    south_states = {
        "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky", "Louisiana", "Maryland",
        "Mississippi", "North Carolina", "Oklahoma", "South Carolina", "Tennessee", "Texas",
        "Virginia", "West Virginia"
    }
    west_states = {
        "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada",
        "New Mexico", "Oregon", "Utah", "Washington", "Wyoming"
    }

    def get_region_for_state(state):
        if state in northeast_states:
            return 'Northeast'
        elif state in midwest_states:
            return 'Midwest'
        elif state in south_states:
            return 'South'
        elif state in west_states:
            return 'West'
        else:
            return 'Other'

    states_codes = list(state_abbrev.values())
    states_names = list(state_abbrev.keys())

    # Reordered region_color_map to match West, Midwest, South, Northeast
    region_color_map = {
        'West': 0,
        'Midwest': 1,
        'South': 2,
        'Northeast': 3,
        'Other': 4
    }

    # Reordered colorscale accordingly
    colorscale = [
        [0.0, '#ab63fa'],   # West - purple
        [0.25, '#ab63fa'],
        [0.25, '#EF553B'],  # Midwest - red
        [0.5, '#EF553B'],
        [0.5, '#00cc96'],   # South - green
        [0.75, '#00cc96'],
        [0.75, '#636efa'],  # Northeast - blue
        [1.0, '#636efa']
    ]

    z = [region_color_map.get(get_region_for_state(st), 4) for st in states_names]

    fig = go.Figure(data=go.Choropleth(
        locations=states_codes,
        z=z,
        locationmode='USA-states',
        colorscale=colorscale,
        showscale=False,
        marker_line_color='white',
        marker_line_width=1,
        zmin=0,
        zmax=3
    ))

    if selected_state_name in state_abbrev:
        selected_code = state_abbrev[selected_state_name]
        fig.add_trace(go.Choropleth(
            locations=[selected_code],
            z=[1],  # dummy value
            locationmode='USA-states',
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # transparent fill
            showscale=False,
            marker_line_color='orange',
            marker_line_width=4,
            hoverinfo='location',
            name='Selected State'
        ))

    fig.update_geos(
        visible=False,   # hides axis lines & ticks
        showcountries=True,
        showlakes=True,
        lakecolor='rgb(255, 255, 255)',
        projection_type='albers usa',
    )

    # Legend order updated
    region_names = ['West', 'Midwest', 'South', 'Northeast']
    region_colors = ['#ab63fa', '#EF553B', '#00cc96', '#636efa']

    legend_annotations = []
    x_start = 0.1
    x_gap = 0.2
    y_pos = -0.15  # Position below the map

    for i, (name, color) in enumerate(zip(region_names, region_colors)):
        x = x_start + i * x_gap
        # Colored box (square)
        legend_annotations.append(dict(
            x=x,
            y=y_pos,
            xref='paper',
            yref='paper',
            showarrow=False,
            text="&nbsp;&nbsp;&nbsp;&nbsp;",  # space for colored box
            bgcolor=color,
            bordercolor='black',
            borderwidth=1,
            borderpad=2,
            font=dict(size=14),
            align='center',
            valign='middle'
        ))
        # Text label next to the box, now white
        legend_annotations.append(dict(
            x=x + 0.05,
            y=y_pos,
            xref='paper',
            yref='paper',
            showarrow=False,
            text=name,
            font=dict(size=14, color='white'),  # <-- white text color here
            align='left',
            valign='middle'
        ))

    fig.update_layout(
        margin=dict(r=10, t=40, l=10, b=80),
        annotations=legend_annotations,
        title_text='US Map Colored by Region with Selected State Highlight',
        title_x=0,  # Left align the title
        plot_bgcolor='rgba(0,0,0,0)',  # transparent background (optional)
        paper_bgcolor='rgba(0,0,0,0)', # transparent paper background (optional)
    )

    return fig

if section == "House Price Predictor":
    st.header("5. Predict House Price")
    st.write("Enter the details below to predict the house price based on property size, bedrooms, bathrooms, region, city type, area type, and city.")

    # Use lowercase keys for mapping to ensure consistent lookup
    city_type_map = {
        "town": 0, "small city": 1, "medium city": 2, "large city": 3, "metropolis": 4
    }
    area_type_map = {
        "rural": 0, "suburban": 1, "urban": 2
    }

    st.subheader("Select Geographic Attributes")

    # Region dropdown
    regions = sorted(region_state_hierarchy.keys())
    selected_region = st.selectbox("Select Region", ["Select Region"] + regions, index=0)

    # State dropdown
    states = sorted(region_state_hierarchy[selected_region].keys()) if selected_region != "Select Region" else []
    selected_state = st.selectbox("Select State", ["Select State"] + states, index=0)

    # Always show map (no else)
    fig = plot_region_map(selected_state if selected_state != "Select State" else None)
    st.plotly_chart(fig, use_container_width=True)

    # Helper to capitalize options for display
    def capitalize_options(options):
        return [opt.title() for opt in options]

    # City Type dropdown with capitalized display options
    city_types_raw = sorted(region_state_hierarchy[selected_region][selected_state].keys()) \
        if selected_region != "Select Region" and selected_state != "Select State" else []
    city_types_display = capitalize_options(city_types_raw)
    city_type_choice = st.selectbox("Select City Type", ["Select City Type"] + city_types_display, index=0)

    # Map back selected display value to lowercase key
    if city_type_choice != "Select City Type":
        selected_city_type_label = city_type_choice.lower()
    else:
        selected_city_type_label = "Select City Type"

    # Area Type dropdown with capitalized display options
    area_types_raw = sorted(
        region_state_hierarchy[selected_region][selected_state].get(selected_city_type_label, {}).keys()
    ) if (selected_region != "Select Region" and selected_state != "Select State" and selected_city_type_label != "Select City Type") else []
    area_types_display = capitalize_options(area_types_raw)
    area_type_choice = st.selectbox("Select Area Type", ["Select Area Type"] + area_types_display, index=0)

    if area_type_choice != "Select Area Type":
        selected_area_type_label = area_type_choice.lower()
    else:
        selected_area_type_label = "Select Area Type"

    # Convert selections to numeric codes for model
    selected_city_type = city_type_map.get(selected_city_type_label, 0)
    selected_area_type = area_type_map.get(selected_area_type_label, 0)

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
        model_features = ['bed', 'bath', 'acre_lot', 'house_size', 'population_2024', 'density',
                          'city_type', 'area_type', 'property_size', 'bed_bath_ratio',
                          'region_Midwest', 'region_Northeast', 'region_South', 'region_West']
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
