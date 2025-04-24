import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
    json_url = "https://github.com/Shamsfathalla/datascience-streamlit/blob/18aafec3b2da7197224722e28fcc15fe8d378c80/region_state_hierarchy.json"
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
        "bed_bath_ratio": MinMaxScaler(),
        "price": MinMaxScaler()
    }
    # Copy the dataset to avoid modifying the original
    df_transform = df.copy()
    # Apply log1p and PowerTransformer to skew_columns (excluding bed and bath)
    skew_columns = ['acre_lot', 'house_size', 'property_size', 'bed_bath_ratio']
    for col in skew_columns:
        if col in df_transform.columns:
            df_transform[col] = np.log1p(df_transform[col])
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            df_transform[col] = pt.fit_transform(df_transform[[col]])
            power_transformers[col] = pt
    # Apply log1p and PowerTransformer to price
    if 'price' in df_transform.columns:
        df_transform['price'] = np.log1p(df_transform['price'])
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        df_transform['price'] = pt.fit_transform(df_transform[['price']])
        power_transformers['price'] = pt
    # Apply MinMaxScaler
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

# Title
st.title("U.S. Housing Market Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", 
                         ["Home", 
                          "Regional Price Differences", 
                          "Bedrooms/Bathrooms Impact", 
                          "House Size by City Type", 
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
             caption="U.S. Housing Market Analysis", use_container_width=True)

# Regional Price Differences section
elif section == "Regional Price Differences":
    st.header("1. How do property prices differ between the different U.S. regions?")
    # Define region names and dummy columns
    region_names = ['Midwest', 'Northeast', 'South', 'West']
    region_columns = ['region_Midwest', 'region_Northeast', 'region_South', 'region_West']
    # Define color palette
    region_colors = ['blue', 'green', 'red', 'purple']
    # Calculate region-based averages
    region_avg_prices = df[region_columns].mul(df['price'], axis=0).sum() / df[region_columns].sum()
    region_avg_population_2024 = df[region_columns].mul(df['population_2024'], axis=0).sum() / df[region_columns].sum()
    region_avg_density = df[region_columns].mul(df['density'], axis=0).sum() / df[region_columns].sum()
    # Set proper indices
    region_avg_prices.index = region_names
    region_avg_population_2024.index = region_names
    region_avg_density.index = region_names
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Plot 1: Average Property Price by Region
    st.subheader("Average Property Price by Region")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=region_names, y=region_avg_prices.values, hue=region_names, palette=region_colors, ax=ax1, legend=False)
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Average Price")
    for i, v in enumerate(region_avg_prices.values):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    st.pyplot(fig1)
    
    # Plot 2: Average Population in 2024 by Region
    st.subheader("Average Population in 2024 by Region")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=region_names, y=region_avg_population_2024.values, hue=region_names, palette=region_colors, ax=ax2, legend=False)
    ax2.set_xlabel("Region")
    ax2.set_ylabel("Average Population")
    for i, v in enumerate(region_avg_population_2024.values):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    st.pyplot(fig2)
    
    # Plot 3: Average Density by Region
    st.subheader("Average Density by Region")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=region_names, y=region_avg_density.values, hue=region_names, palette=region_colors, ax=ax3, legend=False)
    ax3.set_xlabel("Region")
    ax3.set_ylabel("Average Density")
    for i, v in enumerate(region_avg_density.values):
        ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    st.pyplot(fig3)
    
    st.write("""
    ### Key Insights:
    - West Region:
        - Property Price: The West continues to have the highest average property price at approximately $473,869.94. This is likely due to high demand and desirable locations, such as coastal areas and major cities (e.g., Los Angeles, San Francisco), along with robust economic activity.
        - Population and Density: Despite a moderate population density, the high property prices in the West suggest that factors beyond density, like lifestyle preferences, economic opportunities, and desirable locations, contribute significantly to elevated housing costs.
    - Northeast Region:
        - Property Price: The Northeast has a high average property price of around $277,303.81, driven by its high population density and urban development, particularly in cities like New York and Boston. These factors contribute to the demand for real estate and higher prices.
        - Population and Density: The Northeast's high density (5,942.71) reflects its concentration of major cities and economic activity, which typically correlate with higher property values.
    - South Region:
        - Property Price: The South has an average property price of about $296,500.84, which is slightly higher than the Northeast's average. This suggests that while the South may not have the same level of urbanization as the Northeast, it is experiencing growing regional economies and increasing housing demand, contributing to property price increases.
        - Population and Density: With a lower population density (3,302.13) than the Northeast and West, the South's property prices reflect a mix of economic growth and migration patterns. While density is lower, other factors like regional development and growing demand for housing are influencing prices.
    - Midwest Region:
        - Property Price: The Midwest has the lowest average property price at approximately $231,103.74. This reflects lower demand and different market dynamics compared to other regions. The Midwest is characterized by lower population density (2,771.65) and slower economic growth, which impacts housing demand.
        - Population and Density: The Midwest’s relatively low population density and property prices reflect the region’s economic realities, including fewer major metropolitan areas and less competition for housing.
    ### Final Answer:
    - Property Prices: The property prices vary significantly across U.S. regions, with the following ranking from highest to lowest:
        - West (Highest average property price)
        - South
        - Northeast
        - Midwest (Lowest average property price)
    - These differences are driven by a combination of factors, including population density, urban development, economic activity, and regional demand for housing. The West and Northeast, with their high population density and major economic hubs, maintain the highest property prices. The South benefits from regional economic growth and migration trends, placing it in the middle. The Midwest, with its lower density, slower economic growth, and less competition for housing, has the lowest property prices.
    """)

# Bedrooms/Bathrooms Impact section
elif section == "Bedrooms/Bathrooms Impact":
    st.header("2. How does the number of bedrooms and bathrooms affect home prices?")
    # Validate required columns
    required_cols = ['bed', 'bath', 'bed_bath_ratio', 'price']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing columns: " + ", ".join([col for col in required_cols if col not in df.columns]))
        st.stop()
    # Function to add text labels at line intersections
    def add_labels_to_lineplot(ax, data, x_col, y_col):
        x_values = sorted(data[x_col].unique())  # Sort for consistent plotting
        for x in x_values:
            y = data[data[x_col] == x][y_col].mean()  # Calculate mean price for x
            ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=12)
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Plot 1: Bed vs Price
    st.subheader("Bedrooms vs Price")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='bed', y='price', data=df, ax=ax1, marker='o', markersize=8)
    add_labels_to_lineplot(ax1, df, 'bed', 'price')
    ax1.set_xlabel('Number of Bedrooms')
    ax1.set_ylabel('Price')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    st.pyplot(fig1)
    
    # Plot 2: Bath vs Price
    st.subheader("Bathrooms vs Price")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='bath', y='price', data=df, ax=ax2, marker='o', markersize=8)
    add_labels_to_lineplot(ax2, df, 'bath', 'price')
    ax2.set_xlabel('Number of Bathrooms')
    ax2.set_ylabel('Price')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    st.pyplot(fig2)
    
    # Plot 3: Bed_Bath_Ratio vs Price
    st.subheader("Bed/Bath Ratio vs Price")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='bed_bath_ratio', y='price', data=df, ax=ax3, marker='o', markersize=8)
    add_labels_to_lineplot(ax3, df, 'bed_bath_ratio', 'price')
    ax3.set_xlabel('Bed to Bath Ratio')
    ax3.set_ylabel('Price')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.4f}'))
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
    st.pyplot(fig3)
    
    st.write("""
    ### Key Insights:
    - Number of Bedrooms: Increasing the number of bedrooms generally increases the home price, but the marginal increase in price diminishes as the number of bedrooms grows. 
    - Number of Bathrooms: Similarly, increasing the number of bathrooms raises the home price, with diminishing returns as the number of bathrooms increases. 
    - Bed/Bath Ratio: A balanced ratio of bedrooms to bathrooms (around 1.25) tends to maximize home prices. Homes with an imbalanced ratio (too many bedrooms relative to bathrooms or vice versa) may have lower prices.
        - Location and Area Type:
            - Smaller cities and rural areas tend to have fewer bedrooms and bathrooms compared to larger cities and urban areas.
            - Urban areas, despite having more bedrooms and bathrooms, may have slightly lower ratios due to space constraints, but the principle of balance still applies.
    ### Final Answer:
    - The number of bedrooms and bathrooms positively affects home prices, but the impact diminishes as these numbers increase. A balanced bed/bath ratio (around 1.25) is optimal for maximizing home prices. However, it is important to note that bedrooms and bathrooms are not the only factors influencing home prices. These variables are continuous and likely interact with other features (e.g., location, property size, and amenities), which should be considered in a comprehensive pricing model.
    """)

# House Size by City Type section
elif section == "House Size by City Type":
    st.header("3. What is the average house size per city types in the U.S.?")
    # Map area_type and city_type labels
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
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Plot 1: Average House Size by City Type (Line plot)
    st.subheader("Average House Size by City Type")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    avg_house_size_city_type = df.groupby('city_type')['house_size'].mean().reset_index()
    avg_house_size_city_type['city_type_label'] = avg_house_size_city_type['city_type'].map(city_type_labels)
    sns.lineplot(x='city_type_label', y='house_size', data=avg_house_size_city_type, marker='o', ax=ax1)
    ax1.set_xlabel('City Type')
    ax1.set_ylabel('Average House Size (sq ft)')
    # Add exact values to each point
    for i, row in avg_house_size_city_type.iterrows():
        ax1.text(i, row['house_size'], f"{row['house_size']:.4f}", ha='center', va='bottom', fontsize=10)
    st.pyplot(fig1)
    
    # Plot 2: Average Property Size by City Type (Line plot)
    st.subheader("Average Property Size by City Type")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    avg_size_city_type = df.groupby('city_type')['property_size'].mean().reset_index()
    avg_size_city_type['city_type_label'] = avg_size_city_type['city_type'].map(city_type_labels)
    sns.lineplot(x='city_type_label', y='property_size', data=avg_size_city_type, marker='o', ax=ax2)
    ax2.set_xlabel("City Type")
    ax2.set_ylabel("Average Property Size")
    # Add exact values to each point
    for i, row in avg_size_city_type.iterrows():
        ax2.text(i, row['property_size'], f"{row['property_size']:.4f}", ha='center', va='bottom', fontsize=10)
    st.pyplot(fig2)
    
    # Plot 3: Average Acre Lot by City Type (Line plot)
    st.subheader("Average Acre Lot by City Type")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    avg_acre_lot_city_type = df.groupby('city_type')['acre_lot'].mean().reset_index()
    avg_acre_lot_city_type['city_type_label'] = avg_acre_lot_city_type['city_type'].map(city_type_labels)
    sns.lineplot(x='city_type_label', y='acre_lot', data=avg_acre_lot_city_type, marker='o', ax=ax3)
    ax3.set_xlabel('City Type')
    ax3.set_ylabel('Average Acre Lot (acres)')
    # Add exact values to each point
    for i, row in avg_acre_lot_city_type.iterrows():
        ax3.text(i, row['acre_lot'], f"{row['acre_lot']:.4f}", ha='center', va='bottom', fontsize=10)
    st.pyplot(fig3)
    
    st.write("""
    ### Key Insights:
    - House Sizes:
        - Larger in less densely populated areas (towns, small cities, rural, suburban).
        - Smaller in densely populated areas (large cities, metropolises, urban).
    - Property Sizes and Lot Sizes:
        - Larger in rural and suburban areas.
        - Smaller in urban areas.
    - Population Density:
        - Lower in towns and small cities, leading to larger houses and properties.
        - Higher in large cities and metropolises, leading to smaller houses and properties.
    - Bedroom and Bathroom Counts:
        - More bedrooms in less densely populated areas.
        - Fewer bedrooms but relatively stable bathroom counts in densely populated areas.
    ### Conclusion
    - The average house size per city type in the U.S., based on the provided graph, is as follows:
        - Town: 1,721.2669 sq ft
        - Small City: 1,753.2493 sq ft
        - Medium City: 1,692.1574 sq ft
        - Large City: 1,643.5799 sq ft
        - Metropolis: 1,621.2234 sq ft
    ### Final Answer:
    - The trend shows that house sizes are largest in less densely populated areas (towns and small cities) and decrease as population density increases in larger cities and metropolises. This is supported by related graphs showing decreasing property sizes, lot sizes, and increasing population density in more urbanized areas.
    """)

# Urban/Suburban/Rural Prices section
elif section == "Urban/Suburban/Rural Prices":
    st.header("4. How do prices fluctuate between urban, suburban and rural cities?")
    # Mapping for readability
    area_type_map = {0: 'Rural', 1: 'Suburban', 2: 'Urban'}
    city_type_labels = {
        0: 'Town',
        1: 'Small City',
        2: 'Medium City',
        3: 'Large City',
        4: 'Metropolis'
    }
    # Apply readable labels
    df['area_type_label'] = df['area_type'].map(area_type_map)
    df['city_type_label'] = df['city_type'].map(city_type_labels)
    # Convert city_type_label to a categorical data type with the specified order
    city_type_order = ['Town', 'Small City', 'Medium City', 'Large City', 'Metropolis']
    df['city_type_label'] = pd.Categorical(df['city_type_label'], categories=city_type_order, ordered=True)
    # Calculate the necessary statistics
    avg_price = df.groupby('area_type_label')['price'].mean().reset_index()
    avg_property_size = df.groupby('area_type_label')['property_size'].mean().reset_index()
    avg_pop = df.groupby('area_type_label')['population_2024'].mean().reset_index()
    avg_density = df.groupby('area_type_label')['density'].mean().reset_index()
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Plot 1: Average Property Price by Area Type
    st.subheader("Average Property Price by Area Type")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=avg_price, x='area_type_label', y='price', marker='o', ax=ax1)
    for i, row in avg_price.iterrows():
        ax1.text(row['area_type_label'], row['price'], f"{row['price']:.4f}", fontsize=12, ha='center', va='bottom')
    ax1.set_xlabel("Area Type")
    ax1.set_ylabel("Average Price")
    st.pyplot(fig1)
    
    # Plot 2: Average Property Size by Area Type
    st.subheader("Average Property Size by Area Type")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=avg_property_size, x='area_type_label', y='property_size', marker='o', ax=ax2)
    for i, row in avg_property_size.iterrows():
        ax2.text(row['area_type_label'], row['property_size'], f"{row['property_size']:.4f}", fontsize=12, ha='center', va='bottom')
    ax2.set_xlabel("Area Type")
    ax2.set_ylabel("Average Property Size")
    st.pyplot(fig2)
    
    # Plot 3: Average Population (2024) by Area Type
    st.subheader("Average Population (2024) by Area Type")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=avg_pop, x='area_type_label', y='population_2024', marker='o', ax=ax3)
    for i, row in avg_pop.iterrows():
        ax3.text(row['area_type_label'], row['population_2024'], f"{row['population_2024']:.4f}", fontsize=12, ha='center', va='bottom')
    ax3.set_xlabel("Area Type")
    ax3.set_ylabel("Average Population")
    st.pyplot(fig3)
    
    # Plot 4: Average Population Density by Area Type
    st.subheader("Average Population Density by Area Type")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=avg_density, x='area_type_label', y='density', marker='o', ax=ax4)
    for i, row in avg_density.iterrows():
        ax4.text(row['area_type_label'], row['density'], f"{row['density']:.4f}", fontsize=12, ha='center', va='bottom')
    ax4.set_xlabel("Area Type")
    ax4.set_ylabel("Average Density")
    st.pyplot(fig4)
    
    st.write("""
    ### Key Insights:
    - Urban areas have the highest property prices, driven by high population density, limited land availability, and strong demand.
    - Suburban areas fall in the mid-range, offering a balance between affordability and features like more bedrooms, bathrooms, and moderate property sizes.
    - Rural areas have the lowest property prices, featuring larger lots and homes but fewer amenities and lower demand.
    - In summary, property prices are influenced by factors such as population density, property and lot size, and available amenities (e.g., number of bedrooms and bathrooms). Urban areas command premium prices due to space constraints and high demand, rural areas offer spacious properties at lower costs, and suburban areas strike a balance between the two.
    ### Final Answer:
    - Property prices generally increase from rural to suburban to urban areas, primarily due to differences in population density, land availability, property size, and amenities.
    """)

# House Price Predictor section
elif section == "House Price Predictor":
    st.header("5. Predict House Price")
    st.write("Enter the details below to predict the house price based on property size, bedrooms, bathrooms, region, city type, area type, and city.")
    # Map numerical codes to labels
    city_type_map = {
        "Town": 0, "Small City": 1, "Medium City": 2, "Large City": 3, "Metropolis": 4
    }
    area_type_map = {
        "rural": 0, "suburban": 1, "urban": 2
    }
    st.subheader("Select Geographic Attributes")
    # Region dropdown
    regions = list(region_state_hierarchy.keys())
    selected_region = st.selectbox("Select Region", sorted(regions))
    # State dropdown filtered by region
    states = list(region_state_hierarchy[selected_region].keys())
    selected_state = st.selectbox("Select State", sorted(states))
    # City type dropdown filtered by region and state
    city_types = list(region_state_hierarchy[selected_region][selected_state].keys())
    selected_city_type_label = st.selectbox("Select City Type", sorted(city_types))
    selected_city_type = city_type_map.get(selected_city_type_label, 0)
    # Area type dropdown filtered by region, state, and city type
    area_types = list(region_state_hierarchy[selected_region][selected_state][selected_city_type_label].keys())
    selected_area_type_label = st.selectbox("Select Area Type", sorted(area_types))
    selected_area_type = area_type_map.get(selected_area_type_label.lower(), 0)
    # City dropdown with all cities for the selected filters
    st.subheader("Select City")
    try:
        cities = region_state_hierarchy[selected_region][selected_state][selected_city_type_label][selected_area_type_label]
        if not isinstance(cities, list):
            st.error(f"Expected a list of cities for {selected_region} > {selected_state} > {selected_city_type_label} > {selected_area_type_label}, got: {type(cities)}")
            cities = []
    except (KeyError, TypeError) as e:
        st.error(f"Error accessing city list for {selected_region} > {selected_state} > {selected_city_type_label} > {selected_area_type_label}: {str(e)}")
        cities = []
    # Sort cities if available
    cities = sorted(cities) if cities else []
    # Ensure there is at least one option in the dropdown
    if not cities:
        st.warning("No cities available for this selection. Please adjust your filters.")
        cities = ["No cities available"]
    # City dropdown with all available cities
    selected_city = st.selectbox("Select City", cities)
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
