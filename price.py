# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# App Header
st.title("🏠 House Price Predictor")
st.markdown("Enter the details below to predict the price of your house.")

# Create a synthetic dataset with location information
@st.cache_data
def create_enhanced_dataset():
    # Start with the California housing dataset
    housing_data = fetch_california_housing()
    df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    df['PRICE'] = housing_data.target * 100000  # Convert to actual dollar amounts

    # Add synthetic location data (5 different neighborhoods)
    np.random.seed(42)
    locations = ['Downtown', 'Suburbs', 'Rural', 'Coastal', 'Urban']
    df['Location'] = np.random.choice(locations, size=len(df))

    # Rename columns to be more intuitive
    df = df.rename(columns={
        'MedInc': 'Income',
        'HouseAge': 'HouseAge',
        'AveRooms': 'AvgRooms',
        'AveBedrms': 'AvgBedrooms',
        'Population': 'Population',
        'AveOccup': 'AvgOccupancy',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    })

    # Create a simplified dataset with key features
    simplified_df = df[['AvgBedrooms', 'Income', 'HouseAge', 'Location', 'PRICE']].copy()
    simplified_df = simplified_df.rename(columns={'AvgBedrooms': 'Bedrooms'})

    # Convert bedrooms to integers (rounding)
    simplified_df['Bedrooms'] = simplified_df['Bedrooms'].round().astype(int)

    # Create area feature (synthetic based on income and bedrooms)
    simplified_df['Area'] = (simplified_df['Income'] * 500) + (simplified_df['Bedrooms'] * 200) + np.random.normal(0, 200, len(simplified_df))

    # Ensure area is positive
    simplified_df['Area'] = simplified_df['Area'].abs()

    # Select final columns
    final_df = simplified_df[['Area', 'Bedrooms', 'Location', 'PRICE']]

    return final_df

df = create_enhanced_dataset()

# Prepare data for modeling
X = df[['Area', 'Bedrooms', 'Location']]
y = df['PRICE']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create preprocessing pipeline
numeric_features = ['Area', 'Bedrooms']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Location']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train models
@st.cache_resource
def train_best_model(_X_train, _y_train, _X_test, _y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(_X_train, _y_train)
        y_pred = pipeline.predict(_X_test)
        r2 = r2_score(_y_test, y_pred)
        results[name] = {'pipeline': pipeline, 'R2 Score': r2}

    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'R2 Score': [results[model]['R2 Score'] for model in results]
    })
    
    best_model_name = comparison_df.loc[comparison_df['R2 Score'].idxmax(), 'Model']
    return results[best_model_name]['pipeline'], best_model_name, results[best_model_name]['R2 Score']

best_pipeline, best_model_name, best_r2 = train_best_model(X_train, y_train, X_test, y_test)

# Sidebar for model info
with st.sidebar:
    st.header("Model Information")
    st.info(f"**Best Model:** {best_model_name}")
    st.success(f"**Accuracy (R²):** {best_r2:.4f}")

# Function to predict house price based on user input
def predict_house_price(area, bedrooms, location):
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Location': [location]
    })
    prediction = best_pipeline.predict(input_data)[0]
    return prediction

# Main UI layout
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000, step=100)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)

with col2:
    location = st.selectbox("Location", ['Downtown', 'Suburbs', 'Rural', 'Coastal', 'Urban'])

if st.button("Predict Price", type="primary"):
    price = predict_house_price(area, bedrooms, location)
    st.divider()
    st.balloons()
    st.subheader("Results")
    st.metric(label="Predicted House Price", value=f"${price:,.2f}")
    
    st.write(f"**Details:** {bedrooms} Bedroom house in {location} with {area} sq ft.")