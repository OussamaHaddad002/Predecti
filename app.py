import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import os
import io
import base64
import openpyxl
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Predecti !", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .chart-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>Predecti !</h1>", unsafe_allow_html=True)
st.markdown("""
This application uses machine learning to predict diabetes risk based on your health metrics.
You can choose between multiple prediction models or explore your data through clustering analysis.
""")

# Function to download sample data
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="diabetes_dataset_sample.csv">Download Sample Dataset</a>'
    return href

# Load the dataset
@st.cache_data
def load_diabetes_dataset():
    try:
        df = pd.read_csv("diabetes.csv")
        return df
    except:
        # Create a sample dataset if the file doesn't exist
        st.warning("Original dataset not found. Using synthetic data for demonstration.")
        np.random.seed(42)
        data = {
            'Pregnancies': np.random.randint(0, 17, 768),
            'Glucose': np.random.randint(50, 200, 768),
            'BloodPressure': np.random.randint(30, 150, 768),
            'SkinThickness': np.random.randint(0, 100, 768),
            'Insulin': np.random.randint(0, 600, 768),
            'BMI': np.random.uniform(15, 60, 768),
            'DiabetesPedigreeFunction': np.random.uniform(0.07, 2.5, 768),
            'Age': np.random.randint(20, 80, 768),
            'Outcome': np.random.randint(0, 2, 768)
        }
        return pd.DataFrame(data)

# Load saved models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Random Forest': 'diabetes_rf.pkl',
        'Logistic Regression': 'diabetes_lr.pkl',
        'KNN': 'diabetes_knn.pkl',
        'SVC': 'diabetes_svm.pkl'
    }
    
    for name, file in model_files.items():
        try:
            models[name] = pickle.load(open(file, 'rb'))
        except:
            st.warning(f"Model file {file} not found. Some prediction options may be unavailable.")
            
    return models

# Load data and models
df = load_diabetes_dataset()
models = load_models()

# Define sidebar navigation with links instead of dropdown
st.sidebar.markdown("<h2>Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<style>
.nav-link {
    padding: 10px 15px;
    margin: 8px 0;
    text-align: left;
    font-weight: normal;
    text-decoration: none !important;
    color: #000000 !important;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    display: flex;
    align-items: center;
    transition: all 0.3s;
}
.nav-link:hover {
    background-color: rgba(255, 255, 255, 0.2);
    transform: translateX(5px);
    cursor: pointer;
}
.active-page {
    background-color: #4e8df5 !important;
    color: white !important;
    font-weight: bold !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.nav-icon {
    margin-right: 10px;
    font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)

# Icons for each page
icons = {
    "Home": "",
    "Prediction Tool": "",
    "Clustering Analysis": "",
    "File Import": "",
    "About": ""
}

# Get current page from URL parameters or set default
page = st.query_params.get("page", "Home")

# Create navigation links
pages = ["Home", "Prediction Tool", "Clustering Analysis", "File Import", "About"]
for page_name in pages:
    active_class = "active-page" if page_name == page else ""
    icon = icons.get(page_name, "")
    st.sidebar.markdown(
        f"""<a href="?page={page_name}" target="_self" 
        class="nav-link {active_class}"><span class="nav-icon">{icon}</span> {page_name}</a>""", 
        unsafe_allow_html=True
    )

# Page navigation
if page == "Home" or page not in pages:
    
    st.write(""" 
    Use the sidebar to navigate to different sections:
    - **Prediction Tool**: Get personalized diabetes risk prediction
    - **Clustering Analysis**: See how your health metrics compare with others
    - **About**: Learn more about this application and its models
    """)
    
    # Display sample data
    st.markdown("<h3 class='sub-header'>Sample Dataset</h3>", unsafe_allow_html=True)
    st.dataframe(df.sample(10).reset_index(drop=True))
    st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

# Function to add derived features
def add_derived_features(df):
    # Add BMI categories
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obesity 1', 'Obesity 2', 'Obesity 3']
    )
    
    # Add Insulin Score
    df['Insulin_Score'] = pd.cut(
        df['Insulin'],
        bins=[0, 16, 166, float('inf')],
        labels=['Low', 'Normal', 'High']
    )
    
    # Add Glucose Categories
    df['Glucose_Category'] = pd.cut(
        df['Glucose'],
        bins=[0, 70, 99, 126, float('inf')],
        labels=['Low', 'Normal', 'Overweight', 'High']
    )
    
    return df

# Preprocess data for models
def preprocess_for_models(input_data):
    # Extract features
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age = input_data
    
    # Handle zero values which might be placeholders for missing data
    # These corrections improve model accuracy by addressing missing data issues
    if skin_thickness == 0:
        skin_thickness = 20.5  # Mean value from dataset
    
    if insulin == 0:
        insulin = 79.8  # Mean value from dataset
    
    if blood_pressure == 0:
        blood_pressure = 69.1  # Mean value from dataset
    
    # Create DataFrame with the input values
    input_df = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Create the derived categorical features
    # NewBMI categorization
    if bmi < 18.5:
        input_df['NewBMI_Underweight'] = 1
        input_df['NewBMI_Overweight'] = 0
        input_df['NewBMI_Obesity 1'] = 0
        input_df['NewBMI_Obesity 2'] = 0
        input_df['NewBMI_Obesity 3'] = 0
    elif 18.5 <= bmi <= 24.9:
        # Normal weight is the reference category (dropped in one-hot encoding)
        input_df['NewBMI_Underweight'] = 0
        input_df['NewBMI_Overweight'] = 0
        input_df['NewBMI_Obesity 1'] = 0
        input_df['NewBMI_Obesity 2'] = 0
        input_df['NewBMI_Obesity 3'] = 0
    elif 24.9 < bmi <= 29.9:
        input_df['NewBMI_Underweight'] = 0
        input_df['NewBMI_Overweight'] = 1
        input_df['NewBMI_Obesity 1'] = 0
        input_df['NewBMI_Obesity 2'] = 0
        input_df['NewBMI_Obesity 3'] = 0
    elif 29.9 < bmi <= 34.9:
        input_df['NewBMI_Underweight'] = 0
        input_df['NewBMI_Overweight'] = 0
        input_df['NewBMI_Obesity 1'] = 1
        input_df['NewBMI_Obesity 2'] = 0
        input_df['NewBMI_Obesity 3'] = 0
    elif 34.9 < bmi <= 39.9:
        input_df['NewBMI_Underweight'] = 0
        input_df['NewBMI_Overweight'] = 0
        input_df['NewBMI_Obesity 1'] = 0
        input_df['NewBMI_Obesity 2'] = 1
        input_df['NewBMI_Obesity 3'] = 0
    else:  # bmi > 39.9
        input_df['NewBMI_Underweight'] = 0
        input_df['NewBMI_Overweight'] = 0
        input_df['NewBMI_Obesity 1'] = 0
        input_df['NewBMI_Obesity 2'] = 0
        input_df['NewBMI_Obesity 3'] = 1
    
    # NewInsulinScore categorization - improved boundaries
    if insulin < 16:
        input_df['NewInsulinScore_Low'] = 1
        input_df['NewInsulinScore_Normal'] = 0
        input_df['NewInsulinScore_High'] = 0
    elif 16 <= insulin <= 166:
        input_df['NewInsulinScore_Low'] = 0
        input_df['NewInsulinScore_Normal'] = 1
        input_df['NewInsulinScore_High'] = 0
    else:
        input_df['NewInsulinScore_Low'] = 0
        input_df['NewInsulinScore_Normal'] = 0
        input_df['NewInsulinScore_High'] = 1
    
    # NewGlucose categorization
    if glucose <= 70:
        input_df['NewGlucose_Low'] = 1
        input_df['NewGlucose_Normal'] = 0
        input_df['NewGlucose_Overweight'] = 0
        input_df['NewGlucose_High'] = 0
    elif 70 < glucose <= 99:
        input_df['NewGlucose_Low'] = 0
        input_df['NewGlucose_Normal'] = 1
        input_df['NewGlucose_Overweight'] = 0
        input_df['NewGlucose_High'] = 0
    elif 99 < glucose <= 126:
        input_df['NewGlucose_Low'] = 0
        input_df['NewGlucose_Normal'] = 0
        input_df['NewGlucose_Overweight'] = 1
        input_df['NewGlucose_High'] = 0
    else:  # glucose > 126
        input_df['NewGlucose_Low'] = 0
        input_df['NewGlucose_Normal'] = 0
        input_df['NewGlucose_Overweight'] = 0
        input_df['NewGlucose_High'] = 1
    
    # Add interaction features which might help with predictive power
    input_df['BMI_x_Age'] = input_df['BMI'] * input_df['Age'] / 100
    input_df['Glucose_x_BMI'] = input_df['Glucose'] * input_df['BMI'] / 100
    
    # Ensure all expected columns are present in the correct order
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'NewBMI_Underweight', 
        'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
        'NewBMI_Obesity 3', 'NewInsulinScore_Low', 'NewInsulinScore_Normal', 
        'NewInsulinScore_High', 'NewGlucose_Low', 'NewGlucose_Normal', 
        'NewGlucose_Overweight', 'NewGlucose_High', 'BMI_x_Age', 'Glucose_x_BMI'
    ]
    
    # Check if any expected columns are missing and add them with zeros if needed
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure columns are in the right order for model compatibility
    input_df = input_df[expected_columns]
    
    return input_df

# Function to make predictions
def make_prediction(model, input_data):
    processed_input = preprocess_for_models(input_data)
    
    # Extract original features for interpretation
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age = input_data
    
    # Scale numeric features only (first 8 columns)
    numeric_columns = processed_input.iloc[:, :8]
    categorical_columns = processed_input.iloc[:, 8:]
    
    # Create a more robust scaler using training data-like statistics
    # These values are derived from the diabetes dataset analysis
    scaler = StandardScaler()
    scaler.mean_ = np.array([3.8, 120.9, 69.1, 20.5, 79.8, 32.0, 0.47, 33.2])
    scaler.scale_ = np.array([3.4, 32.0, 19.4, 16.0, 115.2, 7.9, 0.33, 11.8])
    
    # Apply scaling to numeric features
    scaled_numerics = scaler.transform(numeric_columns)
    
    # Combine scaled numeric features with categorical features
    scaled_input = np.hstack([scaled_numerics, categorical_columns])
    
    # Check for feature count mismatch
    expected_features = 18  # The number of features the model was trained with
    actual_features = scaled_input.shape[1]
    
    if actual_features > expected_features:
        # If we have more features than the model expects, we need to trim them
        scaled_input = scaled_input[:, :expected_features]
    elif actual_features < expected_features:
        # If we have fewer features than the model expects, raise an error
        raise ValueError(f"Model expects {expected_features} features but input has only {actual_features} features")
    
    prediction = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)[0]
    
    # Create a properly formatted probability array with both classes
    if len(probabilities) == 2:
        # If the model returns probabilities for both classes, use them directly
        probability = probabilities
    else:
        # If the model returns only one probability, calculate the other one
        neg_prob = 1.0 - probabilities
        probability = np.array([neg_prob, probabilities])
    
    # Debug information for explanation
    debug_info = {
        'risk_factors': [],
        'key_features': {},
        'feature_importances': {}
    }
    
    # Identify key risk factors based on medical thresholds
    # Glucose is a significant factor based on the analysis
    if glucose > 126:
        debug_info['risk_factors'].append(f"High Glucose ({glucose} > 126 mg/dL)")
        debug_info['key_features']['glucose'] = f"HIGH ({glucose})"
    elif glucose > 99:
        debug_info['risk_factors'].append(f"Elevated Glucose ({glucose} mg/dL)")
        debug_info['key_features']['glucose'] = f"Elevated ({glucose})"
    else:
        debug_info['key_features']['glucose'] = f"Normal ({glucose})"
    
    # BMI is another significant factor
    if bmi > 30:
        debug_info['risk_factors'].append(f"Obesity (BMI: {bmi:.1f} > 30)")
        debug_info['key_features']['bmi'] = f"HIGH ({bmi:.1f})"
    elif bmi > 25:
        debug_info['risk_factors'].append(f"Overweight (BMI: {bmi:.1f} > 25)")
        debug_info['key_features']['bmi'] = f"Elevated ({bmi:.1f})"
    else:
        debug_info['key_features']['bmi'] = f"Normal ({bmi:.1f})"
    
    # Diabetes pedigree function
    if diabetes_pedigree > 0.8:
        debug_info['risk_factors'].append(f"High Diabetes Pedigree ({diabetes_pedigree:.2f} > 0.8)")
        debug_info['key_features']['diabetes_pedigree'] = f"HIGH ({diabetes_pedigree:.2f})"
    else:
        debug_info['key_features']['diabetes_pedigree'] = f"Normal ({diabetes_pedigree:.2f})"
    
    # Age is a significant factor based on the analysis
    if age > 50:
        debug_info['risk_factors'].append(f"Higher Age Risk ({age} years)")
        debug_info['key_features']['age'] = f"HIGH risk factor ({age})"
    elif age > 40:
        debug_info['risk_factors'].append(f"Moderate Age Risk ({age} years)")
        debug_info['key_features']['age'] = f"Moderate risk factor ({age})"
    else:
        debug_info['key_features']['age'] = f"Lower risk factor ({age})"
    
    # Add feature importance based on model type if available
    if hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        importances = model.feature_importances_
        feature_names = processed_input.columns
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        debug_info['feature_importances'] = {k: float(v) for k, v in sorted_features[:5]}
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        importances = np.abs(model.coef_[0])
        feature_names = processed_input.columns
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        debug_info['feature_importances'] = {k: float(v) for k, v in sorted_features[:5]}
    
    return prediction[0], probability, debug_info

# Run clustering algorithms
def run_clustering(input_data, df, algorithm="KMeans"):
    """
    Run clustering analysis on the input data
    
    Args:
        input_data: User input as a list of feature values
        df: Original dataframe 
        algorithm: Clustering algorithm to use ("KMeans" or "DBSCAN")
    
    Returns:
        cluster_id: The cluster ID the input belongs to
        diabetes_rate: Rate of diabetes in the cluster
        similar_patients: DataFrame with similar patients
        cluster_data: DataFrame with all patients in the same cluster
    """
    try:
        # Prepare the dataset
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        X = df[features].values
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert input to numpy array and scale
        input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
        
        cluster_id = None
        diabetes_rate = None
        similar_patients = None
        cluster_data = None
        
        if algorithm == "KMeans":
            # K-Means clustering (k=4 based on analysis)
            kmeans = KMeans(n_clusters=4, random_state=42)
            kmeans.fit(X_scaled)
            
            # Predict cluster for input data
            cluster_id = kmeans.predict(input_scaled)[0]
            
            # Get all data points in the same cluster
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
            cluster_data = df.iloc[cluster_indices].copy()
            
            # Calculate diabetes rate in the cluster
            diabetes_rate = cluster_data['Outcome'].mean()
            
            # Find similar patients within the cluster using KNN
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=6)  # 1 more than we need (first will be self)
            nn.fit(X_scaled[cluster_indices])
            
            distances, indices = nn.kneighbors(input_scaled)
            
            # Get the similar patients (skip the first one which is self)
            similar_indices = [cluster_indices[i] for i in indices[0][1:6]]
            similar_patients = df.iloc[similar_indices]
            
        elif algorithm == "DBSCAN":
            # DBSCAN clustering (eps=1.0, min_samples=10 based on analysis)
            dbscan = DBSCAN(eps=1.0, min_samples=10)
            labels = dbscan.fit_predict(X_scaled)
            
            # Find the nearest neighbors for input data
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=6)
            nn.fit(X_scaled)
            
            distances, indices = nn.kneighbors(input_scaled)
            
            # Get the similar patients (skip the first one which would be self)
            similar_indices = indices[0][1:6]
            similar_patients = df.iloc[similar_indices]
            
            # Find out which cluster these neighbors belong to (majority vote)
            neighbor_labels = [labels[i] for i in similar_indices]
            
            if -1 in neighbor_labels:  # If any neighbor is an outlier
                # Check if majority are outliers
                if neighbor_labels.count(-1) > len(neighbor_labels) / 2:
                    cluster_id = -1  # Outlier
                else:
                    # Use most common non-outlier label
                    non_outlier_labels = [l for l in neighbor_labels if l != -1]
                    if non_outlier_labels:
                        from collections import Counter
                        cluster_id = Counter(non_outlier_labels).most_common(1)[0][0]
                    else:
                        cluster_id = -1  # Default to outlier
            else:
                from collections import Counter
                cluster_id = Counter(neighbor_labels).most_common(1)[0][0]
            
            # If identified as part of a cluster, get cluster data and diabetes rate
            if cluster_id != -1:
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_data = df.iloc[cluster_indices].copy()
                diabetes_rate = cluster_data['Outcome'].mean()
            else:
                # For outliers, use nearest neighbors to estimate a rate
                diabetes_rate = similar_patients['Outcome'].mean()
        
        return cluster_id, diabetes_rate, similar_patients, cluster_data
    except Exception as e:
        st.error(f"Error in clustering: {e}")
        return None, None, None, None

if page == "Prediction Tool":
    st.markdown("<h2 class='sub-header'>Prediction Tool</h2>", unsafe_allow_html=True)
    
    # Create model selection dropdown
    available_models = list(models.keys())
    if available_models:
        model_choice = st.selectbox("Select Prediction Model:", available_models)
        selected_model = models[model_choice]
    else:
        st.error("No models are available. Please make sure model files are in the current directory.")
        st.stop()
    
    # Input form
    st.markdown("<h3 class='sub-header'>Enter Your Health Metrics</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 4, help="Number of times pregnant")
        glucose = st.slider("Glucose (mg/dL)", 50, 200, 140, help="Plasma glucose concentration")
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 30, 150, 80, help="Diastolic blood pressure")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 30, help="Triceps skin fold thickness")
    with col2:
        insulin = st.slider("Insulin (mu U/ml)", 0, 600, 150, help="2-Hour serum insulin")
        bmi = st.slider("BMI", 15.0, 60.0, 33.0, help="Body mass index")
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.07, 2.5, 0.6, help="Diabetes pedigree function")
        age = st.slider("Age", 20, 80, 45, help="Age in years")
    
    # Display derived features
    st.markdown("<h3 class='sub-header'>Derived Health Categories</h3>", unsafe_allow_html=True)
    # Calculate BMI category
    bmi_category = "Underweight" if bmi < 18.5 else \
                   "Normal" if 18.5 <= bmi <= 24.9 else \
                   "Overweight" if 24.9 < bmi <= 29.9 else \
                   "Obesity Class 1" if 29.9 < bmi <= 34.9 else \
                   "Obesity Class 2" if 34.9 < bmi <= 39.9 else \
                   "Obesity Class 3"
    
    # Calculate Insulin Score
    insulin_score = "Low" if insulin < 16 else \
                    "Normal" if 16 <= insulin <= 166 else "High"
    
    # Calculate Glucose Category
    glucose_category = "Low" if glucose <= 70 else \
                       "Normal" if 70 < glucose <= 99 else \
                       "Overweight" if 99 < glucose <= 126 else \
                       "High"
    
    # Display in a nice grid
    cols = st.columns(3)
    with cols[0]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("BMI Category", bmi_category)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Insulin Score", insulin_score)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Glucose Category", glucose_category)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Collect input data
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                 insulin, bmi, diabetes_pedigree, age]
    
    # Make prediction
    if st.button("Predict Diabetes Risk"):
        prediction, probability, debug_info = make_prediction(selected_model, input_data)
        
        st.markdown("<h3 class='sub-header'>Prediction Results</h3>", unsafe_allow_html=True)
        
        # Create columns for prediction result and probability
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Prediction")
            if prediction == 1:
                st.error("⚠ High Risk of Diabetes")
            else:
                st.success("✅ Low Risk of Diabetes")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Probability")
            prob_df = pd.DataFrame({
                'Class': ['No Diabetes', 'Diabetes'],
                'Probability': [probability[0], probability[1]]
            })
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Class', y='Probability', data=prob_df, ax=ax)
            ax.set_ylim(0, 1)
            for i, v in enumerate(probability):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display model confidence and explanations
        st.markdown("<h3 class='sub-header'>Prediction Explanation</h3>", unsafe_allow_html=True)
        
        # Risk factors section
        if debug_info['risk_factors']:
            st.warning("**Risk Factors Identified:**")
            for factor in debug_info['risk_factors']:
                st.markdown(f"- {factor}")
        else:
            st.info("No major risk factors were identified in your input.")
        
        # Key factors that influenced prediction
        st.markdown("### Key Metrics Analysis")
        st.markdown("These are the key metrics that strongly influence diabetes predictions:")
        feature_cols = st.columns(4)
        with feature_cols[0]:
            st.markdown(f"**Glucose**: {debug_info['key_features']['glucose']}")
        with feature_cols[1]:
            st.markdown(f"**BMI**: {debug_info['key_features']['bmi']}")
        with feature_cols[2]:
            st.markdown(f"**Diabetes Pedigree**: {debug_info['key_features']['diabetes_pedigree']}")
        with feature_cols[3]:
            st.markdown(f"**Age**: {debug_info['key_features']['age']}")
        
        # Model confidence text
        diab_prob = probability[1]
        if diab_prob > 0.7:
            st.error(f"The model is highly confident ({diab_prob:.2%}) you are at risk for diabetes.")
        elif diab_prob > 0.5:
            st.warning(f"The model indicates moderate risk ({diab_prob:.2%}) for diabetes.")
        elif diab_prob > 0.3:
            st.warning(f"The model indicates borderline risk ({diab_prob:.2%}) for diabetes.")
        else:
            st.success(f"The model is confident ({(1-diab_prob):.2%}) you are not at risk for diabetes.")
        
        # Advice based on prediction
        st.markdown("### Recommendations")
        if prediction == 1:
            st.markdown("""
            Based on this high-risk prediction, consider:
            - Consulting with a healthcare professional
            - Getting formal glucose tolerance testing
            - Evaluating your diet and exercise routine
            """)
        else:
            st.markdown("""
            Your risk appears low, but always maintain good habits:
            - Regular exercise
            - Balanced diet
            - Regular medical check-ups
            """)
        
        # Compare with average risk factors
        st.markdown("<h3 class='sub-header'>How Your Metrics Compare</h3>", unsafe_allow_html=True)
        
        # Calculate average values for diabetic and non-diabetic patients
        diabetic_avg = df[df['Outcome'] == 1].mean().round(2)
        non_diabetic_avg = df[df['Outcome'] == 0].mean().round(2)
        
        # Create comparison dataframe
        metrics_comparison = pd.DataFrame({
            'Your Values': input_data,
            'Diabetic Average': [
                diabetic_avg['Pregnancies'], 
                diabetic_avg['Glucose'], 
                diabetic_avg['BloodPressure'],
                diabetic_avg['SkinThickness'], 
                diabetic_avg['Insulin'], 
                diabetic_avg['BMI'],
                diabetic_avg['DiabetesPedigreeFunction'], 
                diabetic_avg['Age']
            ],
            'Non-diabetic Average': [
                non_diabetic_avg['Pregnancies'], 
                non_diabetic_avg['Glucose'], 
                non_diabetic_avg['BloodPressure'],
                non_diabetic_avg['SkinThickness'], 
                non_diabetic_avg['Insulin'], 
                non_diabetic_avg['BMI'],
                non_diabetic_avg['DiabetesPedigreeFunction'], 
                non_diabetic_avg['Age']
            ]
        }, index=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'])
        
        # Display the comparison table
        st.table(metrics_comparison)
        
        # Create visual comparison of key metrics
        st.markdown("<h4 class='sub-header'>Visual Comparison - Key Metrics</h4>", unsafe_allow_html=True)
        
        # Select the most important metrics based on feature importance
        key_metrics = ['Glucose', 'BMI', 'Diabetes Pedigree', 'Age']
        key_indices = [1, 5, 6, 7]  # Corresponding indices in the input_data array
        
        # Create a comparison bar chart - using matplotlib directly for better control
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Custom colors with better contrast
        colors = ["#FF9999", "#FF5555", "#88CC88"]
        
        # Plot each metric in its own subplot
        for i, (metric, idx, ax) in enumerate(zip(key_metrics, key_indices, axes)):
            # Data for this metric
            values = [
                input_data[idx],
                metrics_comparison.loc[metrics_comparison.index[idx], 'Diabetic Average'],
                metrics_comparison.loc[metrics_comparison.index[idx], 'Non-diabetic Average']
            ]
            
            # Categories
            categories = ['Your\nValue', 'Diabetic\nAverage', 'Non-diabetic\nAverage']
            
            # Create the bar chart
            bars = ax.bar(categories, values, color=colors, width=0.6)
            
            # Add values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + (max(values) * 0.02),
                    f'{height:.1f}',
                    ha='center', 
                    va='bottom',
                    fontsize=9
                )
            
            # Set title and customize appearance
            ax.set_title(metric, fontsize=12, pad=10)
            ax.set_ylim(0, max(values) * 1.15)  # Add 15% padding on top
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Make the plot cleaner
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        # Add overall title and adjust layout
        plt.suptitle("Comparison of Key Health Metrics", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Display the plot
        st.pyplot(fig)
        
        # Add percentage difference from diabetic/non-diabetic averages
        st.markdown("<h4 class='sub-header'>How Your Values Compare to Averages</h4>", unsafe_allow_html=True)
        
        # Calculate and display percentage differences for key metrics
        key_metrics_diff = []
        for i, metric in enumerate(key_metrics):
            user_val = input_data[key_indices[i]]
            diabetic_val = metrics_comparison.loc[metrics_comparison.index[key_indices[i]], 'Diabetic Average']
            non_diabetic_val = metrics_comparison.loc[metrics_comparison.index[key_indices[i]], 'Non-diabetic Average']
            
            # Calculate percentage differences
            diff_from_diabetic = ((user_val - diabetic_val) / diabetic_val * 100) if diabetic_val != 0 else 0
            diff_from_non_diabetic = ((user_val - non_diabetic_val) / non_diabetic_val * 100) if non_diabetic_val != 0 else 0
            
            key_metrics_diff.append({
                'Metric': metric, 
                'Your Value': user_val,
                'Diabetic Avg': diabetic_val, 
                'Non-diabetic Avg': non_diabetic_val,
                'Diff from Diabetic': f"{diff_from_diabetic:.1f}%",
                'Diff from Non-diabetic': f"{diff_from_non_diabetic:.1f}%"
            })
        
        # Display the differences
        diff_df = pd.DataFrame(key_metrics_diff)
        st.dataframe(diff_df)
        
        # Add interpretations of the differences
        st.markdown("### Key Insights from Your Metrics")
        
        insights = []
        
        # Glucose interpretation
        user_glucose = input_data[1]
        diabetic_glucose = diabetic_avg['Glucose']
        non_diabetic_glucose = non_diabetic_avg['Glucose']
        
        if user_glucose >= diabetic_glucose:
            insights.append(f"⚠️ **Glucose**: Your glucose level ({user_glucose} mg/dL) is higher than or equal to the average for diabetic patients ({diabetic_glucose} mg/dL).")
        elif user_glucose > non_diabetic_glucose:
            insights.append(f"⚡ **Glucose**: Your glucose level ({user_glucose} mg/dL) is higher than the average for non-diabetic patients ({non_diabetic_glucose} mg/dL).")
        else:
            insights.append(f"✅ **Glucose**: Your glucose level ({user_glucose} mg/dL) is within the range typically seen in non-diabetic patients.")
        
        # BMI interpretation
        user_bmi = input_data[5]
        diabetic_bmi = diabetic_avg['BMI']
        non_diabetic_bmi = non_diabetic_avg['BMI']
        
        if user_bmi >= diabetic_bmi:
            insights.append(f"⚠️ **BMI**: Your BMI ({user_bmi:.1f}) is higher than or equal to the average for diabetic patients ({diabetic_bmi:.1f}).")
        elif user_bmi > non_diabetic_bmi:
            insights.append(f"⚡ **BMI**: Your BMI ({user_bmi:.1f}) is higher than the average for non-diabetic patients ({non_diabetic_bmi:.1f}).")
        else:
            insights.append(f"✅ **BMI**: Your BMI ({user_bmi:.1f}) is within the range typically seen in non-diabetic patients.")
        
        # Display insights
        for insight in insights:
            st.markdown(insight)
        
        # Conclusion based on the combination of risk factors
        risk_count = sum(1 for insight in insights if insight.startswith("⚠️"))
        warning_count = sum(1 for insight in insights if insight.startswith("⚡"))
        
        st.markdown("### Overall Assessment")
        if risk_count > 0:
            st.warning(f"You have {risk_count} high-risk factors that are similar to profiles of patients with diabetes. The prediction model's assessment of {'high' if prediction == 1 else 'low'} risk should be taken seriously.")
        elif warning_count > 0:
            st.info(f"You have {warning_count} metrics in the cautionary range. While the model predicts {'high' if prediction == 1 else 'low'} risk, monitoring these values is recommended.")
        else:
            st.success("Your metrics are generally within healthy ranges compared to typical non-diabetic profiles.")

elif page == "Clustering Analysis":
    st.markdown("<h2 class='sub-header'>Clustering Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This analysis helps identify patterns in the data by grouping similar profiles together. 
    Enter your health metrics to see which group you belong to and compare with others in your cluster.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        clustering_pregnancies = st.slider("Pregnancies (Clustering)", 0, 17, 4, help="Number of times pregnant")
        clustering_glucose = st.slider("Glucose (mg/dL)", 70, 200, 140, help="Plasma glucose concentration")
        clustering_blood_pressure = st.slider("Blood Pressure (mm Hg)", 30, 150, 80, help="Diastolic blood pressure")
        clustering_skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 30, help="Triceps skin fold thickness")
    with col2:
        clustering_insulin = st.slider("Insulin (mu U/ml)", 0, 600, 150, help="2-Hour serum insulin")
        clustering_bmi = st.slider("BMI", 15.0, 60.0, 33.0, help="Body mass index")
        clustering_diabetes_pedigree = st.slider("Diabetes Pedigree", 0.07, 2.5, 0.6, help="Diabetes pedigree function")
        clustering_age = st.slider("Age", 20, 80, 45, help="Age in years")
    
    algorithm = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN"])
    
    if st.button("Run Clustering Analysis"):
        with st.spinner("Analyzing your data..."):
            # Create input data array
            input_data = [
                clustering_pregnancies, clustering_glucose, clustering_blood_pressure, 
                clustering_skin_thickness, clustering_insulin, clustering_bmi,
                clustering_diabetes_pedigree, clustering_age
            ]
            
            # Run clustering
            cluster_id, diabetes_rate, similar_patients, cluster_data = run_clustering(input_data, df, algorithm)
            
            st.markdown("<h3 class='sub-header'>Clustering Results</h3>", unsafe_allow_html=True)
            
            # Display cluster information
            if algorithm == "KMeans":
                cols = st.columns([2, 1])
                with cols[0]:
                    st.markdown(f"### You belong to Cluster {cluster_id + 1}")
                    if diabetes_rate is not None:
                        st.metric("Diabetes Rate in Your Cluster", f"{diabetes_rate*100:.1f}%")
                        if diabetes_rate > 0.5:
                            st.warning(f"⚠️ Your cluster has a high diabetes rate ({diabetes_rate*100:.1f}%)")
                        else:
                            st.success(f"✅ Your cluster has a low diabetes rate ({diabetes_rate*100:.1f}%)")
                
                with cols[1]:
                    if cluster_data is not None:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.pie([cluster_data['Outcome'].value_counts().get(0, 0), 
                               cluster_data['Outcome'].value_counts().get(1, 0)], 
                               labels=["Non-diabetic", "Diabetic"],
                               autopct='%1.1f%%',
                               colors=['green', 'red'],
                               startangle=90)
                        ax.set_title(f"Cluster {cluster_id + 1} Health Status")
                        st.pyplot(fig)
            elif algorithm == "DBSCAN":
                if cluster_id == -1:
                    st.warning("⚠️ Your profile appears to be unique (outlier) and doesn't belong to any cluster.")
                    
                    st.markdown("### Similar Profiles (Based on Nearest Neighbors)")
                    if similar_patients is not None:
                        st.dataframe(similar_patients[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']])
                else:
                    st.success(f"You belong to Cluster {cluster_id + 1}")
                    if diabetes_rate is not None:
                        st.metric("Diabetes Rate in Your Cluster", f"{diabetes_rate*100:.1f}%")
                        
                        cols = st.columns([2, 1])
                        with cols[0]:
                            if diabetes_rate > 0.5:
                                st.warning(f"⚠️ Your cluster has a high diabetes rate ({diabetes_rate*100:.1f}%)")
                            else:
                                st.success(f"✅ Your cluster has a low diabetes rate ({diabetes_rate*100:.1f}%)")
                        
                        with cols[1]:
                            if cluster_data is not None:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.pie([cluster_data['Outcome'].value_counts().get(0, 0), 
                                       cluster_data['Outcome'].value_counts().get(1, 0)], 
                                       labels=["Non-diabetic", "Diabetic"],
                                       autopct='%1.1f%%',
                                       colors=['green', 'red'],
                                       startangle=90)
                                ax.set_title(f"Cluster {cluster_id + 1} Health Status")
                                st.pyplot(fig)
            
            # Compare input metrics with cluster averages
            if cluster_data is not None:
                st.markdown("<h3 class='sub-header'>Your Health Metrics vs. Cluster Average</h3>", unsafe_allow_html=True)
                
                # Calculate cluster averages
                cluster_avg = cluster_data.mean().round(2)
                
                # Your input + healthy and diabetic averages
                comparison_df = pd.DataFrame({
                    'Your Values': input_data,
                    'Cluster Average': [
                        cluster_avg['Pregnancies'], 
                        cluster_avg['Glucose'], 
                        cluster_avg['BloodPressure'], 
                        cluster_avg['SkinThickness'], 
                        cluster_avg['Insulin'], 
                        cluster_avg['BMI'], 
                        cluster_avg['DiabetesPedigreeFunction'], 
                        cluster_avg['Age']
                    ],
                    'Diabetic Average': [
                        df[df['Outcome'] == 1]['Pregnancies'].mean(), 
                        df[df['Outcome'] == 1]['Glucose'].mean(),
                        df[df['Outcome'] == 1]['BloodPressure'].mean(), 
                        df[df['Outcome'] == 1]['SkinThickness'].mean(),
                        df[df['Outcome'] == 1]['Insulin'].mean(), 
                        df[df['Outcome'] == 1]['BMI'].mean(),
                        df[df['Outcome'] == 1]['DiabetesPedigreeFunction'].mean(), 
                        df[df['Outcome'] == 1]['Age'].mean()
                    ],
                    'Non-diabetic Average': [
                        df[df['Outcome'] == 0]['Pregnancies'].mean(), 
                        df[df['Outcome'] == 0]['Glucose'].mean(),
                        df[df['Outcome'] == 0]['BloodPressure'].mean(), 
                        df[df['Outcome'] == 0]['SkinThickness'].mean(),
                        df[df['Outcome'] == 0]['Insulin'].mean(), 
                        df[df['Outcome'] == 0]['BMI'].mean(),
                        df[df['Outcome'] == 0]['DiabetesPedigreeFunction'].mean(), 
                        df[df['Outcome'] == 0]['Age'].mean()
                    ]
                }, index=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                        'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'])
                
                # Round to 2 decimal places
                comparison_df = comparison_df.round(2)
                
                st.table(comparison_df)
                
                # Plot radar chart of user vs averages
                st.markdown("<h3 class='sub-header'>Profile Comparison</h3>", unsafe_allow_html=True)
                try:
                    # Normalize the data for radar chart
                    scaler = StandardScaler()
                    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                    
                    # Scale the data
                    df_scaled = pd.DataFrame(
                        scaler.fit_transform(df[features]), 
                        columns=features
                    )
                    # Calculate averages of scaled data
                    diabetic_avg = df_scaled[df['Outcome'] == 1].mean()
                    non_diabetic_avg = df_scaled[df['Outcome'] == 0].mean()
                    
                    # Scale user input
                    user_input_df = pd.DataFrame([input_data], columns=features)
                    user_scaled = scaler.transform(user_input_df)[0]
                    
                    # Create radar chart
                    categories = ['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'Pedigree', 'Age']
                    N = len(categories)
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, polar=True)
                    
                    # What will be the angle of each axis in the plot (divide the plot / number of variables)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop
                    
                    # Add the first element at the end for closure
                    user_values = list(user_scaled)
                    user_values += user_values[:1]
                    diabetic_values = list(diabetic_avg)
                    diabetic_values += diabetic_values[:1]
                    non_diabetic_values = list(non_diabetic_avg)
                    non_diabetic_values += non_diabetic_values[:1]
                    
                    # Draw the polygon and fill it
                    ax.plot(angles, user_values, linewidth=1, linestyle='solid', label='You')
                    ax.fill(angles, user_values, alpha=0.1)
                    ax.plot(angles, diabetic_values, linewidth=1, linestyle='solid', label='Diabetic Avg')
                    ax.fill(angles, diabetic_values, alpha=0.1)
                    ax.plot(angles, non_diabetic_values, linewidth=1, linestyle='solid', label='Non-diabetic Avg')
                    ax.fill(angles, non_diabetic_values, alpha=0.1)
                    
                    # Add categories
                    plt.xticks(angles[:-1], categories)
                    
                    # Add legend
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not create radar chart: {e}")
                
                # Similar patients section
                st.markdown("<h3 class='sub-header'>Similar Profiles in Your Cluster</h3>", unsafe_allow_html=True)
                if similar_patients is not None:
                    st.dataframe(similar_patients[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']])
                    
                    # Count diabetics in similar patients
                    diabetic_count = similar_patients['Outcome'].sum()
                    if diabetic_count > 2:  # More than 2 out of 5 are diabetic
                        st.warning(f"⚠️ {diabetic_count} out of {len(similar_patients)} similar profiles have diabetes.")
                    else:
                        st.success(f"✅ Only {diabetic_count} out of {len(similar_patients)} similar profiles have diabetes.")
                else:
                    st.warning("No similar profiles found.")
            else:
                st.warning("Could not analyze cluster data. Try a different algorithm or input values.")
            
            # Conclusion and recommendations
            st.markdown("<h3 class='sub-header'>Key Insights</h3>", unsafe_allow_html=True)
            
            # Generate insights based on the clustering results
            insights = []
            if diabetes_rate is not None:
                if diabetes_rate > 0.6:
                    insights.append("⚠️ You belong to a high-risk cluster with significant diabetes prevalence.")
                elif diabetes_rate > 0.4:
                    insights.append("⚡ You belong to a moderate-risk cluster where diabetes is common but not dominant.")
                else:
                    insights.append("✅ You belong to a low-risk cluster where diabetes is less common.")
            
            if cluster_data is not None:
                user_glucose = clustering_glucose
                cluster_glucose = cluster_avg['Glucose']
                if user_glucose > cluster_glucose + 20:
                    insights.append(f"⚠️ Your glucose level ({user_glucose}) is significantly higher than your cluster average ({cluster_glucose:.1f}).")
                
                user_bmi = clustering_bmi
                cluster_bmi = cluster_avg['BMI']
                if user_bmi > cluster_bmi + 5:
                    insights.append(f"⚠️ Your BMI ({user_bmi:.1f}) is higher than your cluster average ({cluster_bmi:.1f}).")
                
                if clustering_age > 40 and clustering_bmi > 30 and clustering_glucose > 126:
                    insights.append("⚠️ Multiple risk factors detected: Age > 40, BMI > 30, and Glucose > 126.")
            
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("No specific insights were generated based on the clustering analysis.")
                
            st.markdown("""
            ### Recommendations
            
            - Use these insights along with the Prediction Tool for a more comprehensive risk assessment
            - Discuss results with healthcare professionals if concerning patterns appear
            - Remember that clustering only shows similarity patterns, not definitive diagnoses
            """)
            
            # Print button to save results
            if st.button("Print Results"):
                st.markdown("### How to save these results")
                st.info("Use your browser's print function (Ctrl+P or Cmd+P) to save this page as a PDF.")
                st.markdown("---")
        
    else:
        st.info("Enter your health metrics and click 'Run Clustering Analysis' to see your results.")

elif page == "File Import":
    st.markdown("<h2 class='sub-header'>File Import and Batch Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This tool allows you to upload CSV or Excel files containing patient data for bulk analysis.
    Upload your file, select a prediction model, and get diabetes risk predictions for all patients.
    """)
    
    # Define column requirements
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    st.info(f"Your file must contain the following columns: {', '.join(required_columns)}")
    
    # File upload widget
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Excel file
                data = pd.read_excel(uploaded_file)
            
            # Check if required columns are present
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success(f"File successfully uploaded! Found {len(data)} records.")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Data Preview", "Batch Prediction", "Visualizations"])
                
                with tab1:
                    st.markdown("<h3 class='sub-header'>Data Preview</h3>", unsafe_allow_html=True)
                    st.dataframe(data.head(10).style.highlight_null())
                    
                    # Display data summary
                    st.markdown("<h4 class='sub-header'>Data Summary</h4>", unsafe_allow_html=True)
                    st.write(f"Total records: {len(data)}")
                    
                    # Check for missing values
                    missing_values = data[required_columns].isnull().sum()
                    if missing_values.sum() > 0:
                        st.warning("Missing values detected in the data:")
                        st.write(missing_values[missing_values > 0])
                    else:
                        st.success("No missing values in required columns.")
                    
                    # Display basic statistics
                    st.markdown("<h4 class='sub-header'>Statistical Summary</h4>", unsafe_allow_html=True)
                    st.dataframe(data[required_columns].describe().style.highlight_max(axis=0))
                
                with tab2:
                    st.markdown("<h3 class='sub-header'>Batch Prediction</h3>", unsafe_allow_html=True)
                    
                    # Model selection
                    available_models = list(models.keys())
                    if available_models:
                        batch_model_choice = st.selectbox("Select Prediction Model for Batch Processing:", available_models)
                        selected_model = models[batch_model_choice]
                        
                        # Predict button
                        if st.button("Run Batch Prediction"):
                            with st.spinner("Processing batch predictions..."):
                                # Create a copy of the dataframe for predictions
                                prediction_df = data.copy()
                                
                                # Process each row for prediction
                                predictions = []
                                probabilities = []
                                
                                # Handle missing values with means
                                for col in required_columns:
                                    if prediction_df[col].isnull().sum() > 0:
                                        prediction_df[col].fillna(prediction_df[col].mean(), inplace=True)
                                
                                # Make predictions for each row
                                for _, row in prediction_df.iterrows():
                                    row_data = [
                                        row['Pregnancies'], row['Glucose'], row['BloodPressure'], 
                                        row['SkinThickness'], row['Insulin'], row['BMI'], 
                                        row['DiabetesPedigreeFunction'], row['Age']
                                    ]
                                    
                                    try:
                                        prediction, probability, _ = make_prediction(selected_model, row_data)
                                        predictions.append(prediction)
                                        probabilities.append(probability[1])  # Probability of diabetes
                                    except Exception as e:
                                        st.error(f"Error making prediction: {e}")
                                        predictions.append(None)
                                        probabilities.append(None)
                                
                                # Add predictions to dataframe
                                prediction_df['Prediction'] = predictions
                                prediction_df['Risk_Probability'] = probabilities
                                prediction_df['Risk_Level'] = pd.cut(
                                    prediction_df['Risk_Probability'],
                                    bins=[0, 0.3, 0.6, 1.0],
                                    labels=['Low', 'Medium', 'High']
                                )
                                
                                # Calculate overall statistics
                                high_risk_count = sum(1 for p in predictions if p == 1)
                                high_risk_percentage = high_risk_count / len(predictions) * 100
                                
                                # Display results
                                st.markdown("<h4 class='sub-header'>Prediction Results</h4>", unsafe_allow_html=True)
                                
                                # Summary metrics in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Analyzed", len(predictions))
                                with col2:
                                    st.metric("High Risk Count", high_risk_count)
                                with col3:
                                    st.metric("High Risk Percentage", f"{high_risk_percentage:.1f}%")
                                
                                # Show prediction results table
                                st.markdown("<h4 class='sub-header'>Detailed Predictions</h4>", unsafe_allow_html=True)
                                
                                # Color code risk levels
                                def color_risk(val):
                                    if val == 'High':
                                        return 'background-color: #ffcccc'
                                    elif val == 'Medium':
                                        return 'background-color: #ffffcc'
                                    elif val == 'Low':
                                        return 'background-color: #ccffcc'
                                    return ''
                                
                                # Display the dataframe with colored risk levels
                                display_cols = required_columns + ['Prediction', 'Risk_Probability', 'Risk_Level']
                                styled_df = prediction_df[display_cols].style.applymap(
                                    color_risk, subset=['Risk_Level']
                                ).format({
                                    'Risk_Probability': '{:.2%}'
                                })
                                st.dataframe(styled_df)
                                
                                # Create a downloadable CSV of predictions
                                csv = prediction_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="batch_prediction_results.csv">Download Prediction Results (CSV)</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                                # Save to session state for the visualization tab
                                st.session_state['prediction_df'] = prediction_df
                    else:
                        st.error("No models are available. Please make sure model files are in the current directory.")
                
                with tab3:
                    st.markdown("<h3 class='sub-header'>Data Visualizations</h3>", unsafe_allow_html=True)
                    
                    # Check if prediction data is available in session state
                    if 'prediction_df' in st.session_state:
                        prediction_df = st.session_state['prediction_df']
                        
                        # Create visualizations
                        st.markdown("<h4 class='sub-header'>Risk Distribution</h4>", unsafe_allow_html=True)
                        
                        # Risk distribution pie chart
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        risk_counts = prediction_df['Risk_Level'].value_counts()
                        ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
                                colors=['green', 'gold', 'red'] if 'Low' in risk_counts.index else ['gold', 'red'] if 'Medium' in risk_counts.index else ['red'],
                                startangle=90)
                        ax1.axis('equal')
                        ax1.set_title('Risk Level Distribution')
                        st.pyplot(fig1)
                        
                        # Risk factors analysis
                        st.markdown("<h4 class='sub-header'>Key Risk Factors Analysis</h4>", unsafe_allow_html=True)
                        
                        # Create feature correlation with risk probability
                        corr_df = prediction_df[required_columns + ['Risk_Probability']]
                        correlations = corr_df.corr()['Risk_Probability'].sort_values(ascending=False)
                        
                        # Display top correlations
                        st.write("Factors most correlated with diabetes risk:")
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        correlations = correlations.drop('Risk_Probability')  # Remove self-correlation
                        sns.barplot(x=correlations.values, y=correlations.index, ax=ax2)
                        ax2.set_title('Risk Factor Correlation with Diabetes Risk')
                        ax2.set_xlabel('Correlation Coefficient')
                        st.pyplot(fig2)
                        
                        # Feature comparison by risk level
                        st.markdown("<h4 class='sub-header'>Feature Distribution by Risk Level</h4>", unsafe_allow_html=True)
                        
                        # Select feature to visualize
                        feature_to_viz = st.selectbox(
                            "Select Feature to Visualize:", 
                            options=required_columns
                        )
                        
                        # Box plot of selected feature by risk level
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x='Risk_Level', y=feature_to_viz, data=prediction_df, ax=ax3)
                        ax3.set_title(f'{feature_to_viz} Distribution by Risk Level')
                        st.pyplot(fig3)
                        
                        # Scatter plot of glucose vs BMI colored by risk level
                        st.markdown("<h4 class='sub-header'>Glucose vs BMI by Risk Level</h4>", unsafe_allow_html=True)
                        
                        fig4, ax4 = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(
                            x='Glucose', 
                            y='BMI', 
                            hue='Risk_Level',
                            palette={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                            data=prediction_df,
                            ax=ax4
                        )
                        ax4.set_title('Glucose vs BMI by Risk Level')
                        st.pyplot(fig4)
                        
                    else:
                        st.info("Run the batch prediction first to see visualizations.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your file has the required columns and proper formatting.")
    else:
        # No file uploaded, show sample format
        st.markdown("<h3 class='sub-header'>Example File Format</h3>", unsafe_allow_html=True)
        
        # Create a sample dataframe
        sample_data = pd.DataFrame({
            'Pregnancies': [2, 5, 0, 1],
            'Glucose': [120, 166, 95, 105],
            'BloodPressure': [80, 72, 65, 70],
            'SkinThickness': [30, 19, 25, 18],
            'Insulin': [95, 175, 65, 85],
            'BMI': [25.4, 33.7, 21.8, 27.2],
            'DiabetesPedigreeFunction': [0.52, 0.81, 0.35, 0.48],
            'Age': [35, 51, 28, 42]
        })
        
        # Display the sample
        st.dataframe(sample_data)
        
        # Provide downloadable sample
        csv = sample_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_upload_format.csv">Download Sample CSV Format</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Tips for preparing data
        st.markdown("<h3 class='sub-header'>Tips for Preparing Your Data</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. Ensure your file includes all required columns with the exact names shown above
        2. Use numeric values for all fields
        3. For CSV files, use comma as the delimiter
        4. For Excel files, place your data in the first worksheet
        5. Handle missing values before uploading, or they will be replaced with average values
        """)

elif page == "About":
    st.markdown("<h2 class='sub-header'>About Predecti</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Understanding the Models
    
    This application uses four different machine learning algorithms to predict diabetes risk:
    
    1. **Random Forest**: An ensemble method that builds multiple decision trees and merges their predictions.
       - Strengths: High accuracy, handles non-linear data well, less prone to overfitting
       - Ideal for: Complex datasets with many features
       
    2. **Logistic Regression**: A statistical model that uses a logistic function to model binary outcomes.
       - Strengths: Simple, interpretable, works well with linearly separable data
       - Ideal for: Understanding feature importance and direction of influence
       
    3. **K-Nearest Neighbors (KNN)**: Classifies based on the majority class of the k nearest data points.
       - Strengths: Simple, no assumptions about data distribution
       - Ideal for: Small to medium datasets with clear boundaries between classes
       
    4. **Support Vector Machine (SVM)**: Creates a hyperplane that best separates the classes.
       - Strengths: Effective in high-dimensional spaces, versatile through kernel functions
       - Ideal for: Complex classification tasks with clear margins of separation
    """)
    
    st.markdown("### Feature Importance")
    
    # Get feature importance from Random Forest model if available
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        if hasattr(rf_model, 'feature_importances_'):
            st.write("Below is the feature importance from the Random Forest model, which shows how influential each factor is in predicting diabetes risk:")
            
            # Get feature names based on model expectations
            basic_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': basic_features,
                'Importance': rf_model.feature_importances_[:len(basic_features)]
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
            ax.set_title('Feature Importance for Diabetes Prediction')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            
            st.pyplot(fig)
            
            # Explanation of top features
            st.markdown("### Explanation of Key Features")
            st.markdown("""
            - **Glucose**: High blood glucose levels are directly linked to diabetes. Values consistently above 126 mg/dL are particularly concerning.
            - **BMI (Body Mass Index)**: Higher BMI values, especially above 30 (obese range), increase diabetes risk significantly.
            - **Age**: Diabetes risk increases with age, particularly after 40 years.
            - **Diabetes Pedigree Function**: This represents the genetic influence and family history. Higher values indicate stronger genetic predisposition.
            """)
    
    # About the dataset
    st.markdown("### About the Dataset")
    st.write("""
    This application uses the Pima Indians Diabetes Database, which contains medical predictor variables for 768 patients, including:
    
    - Number of pregnancies
    - Glucose concentration in plasma 
    - Blood pressure
    - Skin thickness
    - Insulin level
    - BMI (Body Mass Index)
    - Diabetes pedigree function
    - Age
    
    The dataset was originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases.
    """)
    
    # Display sample statistics
    st.markdown("### Dataset Statistics")
    
    # Calculate statistics
    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Median': df.median(),
        'Min': df.min(),
        'Max': df.max(),
        'Std Dev': df.std()
    }).round(2)
    
    st.dataframe(stats_df)
    
    # Information about the app
    st.markdown("### About the Application")
    st.write("""
    This application was developed to demonstrate the use of machine learning in healthcare predictive analytics. 
    
    **Important Note**: This tool is for educational purposes only and should not be used for actual medical diagnosis. 
    Always consult with healthcare professionals for proper medical advice and diagnosis.
    """)
    
    # Load and display ROC curve if available
    try:
        roc_image = plt.imread("roc_diabetes.jpeg")
        st.markdown("### Model Performance")
        st.write("The ROC (Receiver Operating Characteristic) curve below shows the performance of our models:")
        st.image(roc_image, caption="ROC Curve for Diabetes Prediction Models", use_column_width=True)
    except:
        pass

