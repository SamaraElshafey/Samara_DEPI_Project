
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# STEP 1: SET UP THE STREAMLIT APP STRUCTURE
# ==========================================
def main():
    # Configure the page layout and title
    st.set_page_config(
        page_title="Diabetes Prediction Dashboard",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add a title and description
    st.title("Diabetes Risk Prediction Dashboard")
    st.markdown("""
    This application predicts the risk of diabetes based on various health metrics.
    Use the sidebar to navigate through different sections of the application.
    """)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Make Prediction", "Model Information", "Dataset Exploration"]
    )
    
    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Make Prediction":
        prediction_page()
    elif page == "Model Information":
        model_info_page()
    elif page == "Dataset Exploration":
        data_exploration_page()

# STEP 2: CREATE THE HOME PAGE
# ============================
def home_page():
    st.header("Welcome to the Diabetes Prediction Dashboard")
    
    st.write("""
    ### About This Application
    
    This dashboard allows healthcare professionals and individuals to predict diabetes risk 
    based on various health parameters. The prediction model is trained on a comprehensive
    dataset with features like hemoglobin levels, cholesterol, BMI, and more.
    
    ### How to Use This Dashboard
    
    1. Navigate to the **Make Prediction** page to enter health metrics and get a diabetes risk assessment
    2. Explore the **Dataset Exploration** page to understand the relationships between different health factors
    3. Check the **Model Information** page to learn about the machine learning model powering this application
    
    ### Important Disclaimer
    
    This tool is for informational purposes only and should not replace professional medical advice. 
    Always consult with a healthcare provider for proper diagnosis and treatment.
    """)
    
    # Display some key statistics or visual elements
    st.subheader("Diabetes Risk Factors")
    
    # Create three columns for key information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**BMI**\n\nHigher BMI values are associated with increased diabetes risk")
    
    with col2:
        st.info("**Age**\n\nDiabetes risk typically increases with age")
    
    with col3:
        st.info("**Family History**\n\nGenetic factors play a significant role in diabetes risk")
    
    # Add a visual element
    st.image("https://via.placeholder.com/800x400?text=Diabetes+Risk+Factors+Visualization", 
             caption="Illustration of diabetes risk factors", use_column_width=True)

# STEP 3: CREATE THE DATA LOADING AND MODEL FUNCTIONS
# ==================================================
@st.cache_data
def load_data():
    """Load and cache the diabetes dataset"""
    # In a real application, you would load your actual dataset
    # This is placeholder code - replace with your actual data loading code
    
    # For demonstration purposes, creating a sample dataset that matches your schema
    try:
        # Try to load data from a CSV file if it exists
        data = pd.read_csv('diabetes_data.csv')
    except FileNotFoundError:
        # If file doesn't exist, create sample data
        st.warning("Sample data is being used. Replace with your actual dataset.")
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data that matches your schema
        data = pd.DataFrame({
            'Diabetes_State': np.random.randint(0, 2, n_samples),
            'HB': np.random.normal(5.7, 1.0, n_samples),
            'Cholesterol': np.random.normal(200, 40, n_samples),
            'BMI': np.random.normal(26, 5, n_samples),
            'Heart_Disease': np.random.randint(0, 2, n_samples),
            'PhysActivity': np.random.randint(0, 2, n_samples),
            'PhysHlth': np.random.randint(0, 31, n_samples),
            'Gender': np.random.randint(0, 2, n_samples),
            'Age': np.random.randint(18, 90, n_samples),
            'Stroke': np.random.randint(0, 2, n_samples),
            'GenHlth': np.random.randint(1, 6, n_samples),
            'CholCheck': np.random.randint(0, 2, n_samples),
            'Smoker': np.random.randint(0, 2, n_samples),
            'Fruits': np.random.randint(0, 2, n_samples),
            'Veggies': np.random.randint(0, 2, n_samples),
            'HvyAlcoholConsump': np.random.randint(0, 2, n_samples),
            'MentHlth': np.random.randint(0, 31, n_samples),
            'DiffWalk': np.random.randint(0, 2, n_samples)
        })
        
        # Make correlations more realistic
        # People with higher BMI more likely to have diabetes
        high_bmi_idx = data['BMI'] > 30
        data.loc[high_bmi_idx, 'Diabetes_State'] = np.random.choice([0, 1], size=sum(high_bmi_idx), p=[0.3, 0.7])
        
        # Older people more likely to have diabetes
        older_idx = data['Age'] > 60
        data.loc[older_idx, 'Diabetes_State'] = np.random.choice([0, 1], size=sum(older_idx), p=[0.4, 0.6])
    
    return data

@st.cache_resource
def load_or_train_model(data):
    """Load a pre-trained model or train a new one if it doesn't exist"""
    model_path = 'diabetes_model.pkl'
    
    try:
        # Try to load pre-trained model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("Loaded pre-trained model successfully!")
    except FileNotFoundError:
        # If model doesn't exist, train a new one
        st.info("Training new model...")
        
        # Prepare data
        X = data.drop('Diabetes_State', axis=1)
        y = data['Diabetes_State']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")
    
    return model

# STEP 4: CREATE THE PREDICTION PAGE
# =================================
def prediction_page():
    st.header("Diabetes Risk Prediction")
    st.write("Enter your health metrics below to get a prediction of your diabetes risk.")
    
    data = load_data()  # Load data for feature statistics if needed
    model = load_or_train_model(data)  # Load or train model
    
    # Create a form for user inputs
    with st.form("prediction_form"):
        # Create columns for more compact layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hb = st.number_input("Hemoglobin (HB)", 
                                min_value=3.0, max_value=20.0, 
                                value=7.0, step=0.1,
                                help="Normal range is typically 4.5 to 9.5")
            
            cholesterol = st.number_input("Cholesterol", 
                                        min_value=50, max_value=500, 
                                        value=180, step=5,
                                        help="Measured in mg/dL. Normal range is below 200")
            
            bmi = st.number_input("BMI (Body Mass Index)", 
                                min_value=10.0, max_value=50.0, 
                                value=24.5, step=0.5,
                                help="Normal range is 18.5 to 24.9")
            
            heart_disease = st.selectbox("Heart Disease", 
                                        options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes",
                                        help="Do you have a history of heart disease?")
            
            phys_activity = st.selectbox("Physical Activity", 
                                        options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes",
                                        help="Do you engage in regular physical activity?")
            
            phys_hlth = st.slider("Physical Health Issues (days/month)", 
                                min_value=0, max_value=30, 
                                value=0, step=1,
                                help="Number of days in the past month with physical health issues")
        
        with col2:
            gender = st.selectbox("Gender", 
                                options=[0, 1],
                                format_func=lambda x: "Female" if x == 0 else "Male")
            
            age = st.slider("Age", 
                          min_value=18, max_value=100, 
                          value=45, step=1)
            
            stroke = st.selectbox("History of Stroke", 
                                options=[0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes")
            
            gen_hlth = st.slider("General Health", 
                               min_value=1, max_value=5, 
                               value=2, step=1,
                               help="1=Excellent, 5=Poor")
            
            chol_check = st.selectbox("Cholesterol Check in Past 5 Years", 
                                    options=[0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
            
            smoker = st.selectbox("Smoker", 
                                options=[0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            fruits = st.selectbox("Regular Fruit Consumption", 
                                options=[0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes",
                                help="Do you eat fruit at least once a day?")
            
            veggies = st.selectbox("Regular Vegetable Consumption", 
                                 options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 help="Do you eat vegetables at least once a day?")
            
            alcohol = st.selectbox("Heavy Alcohol Consumption", 
                                 options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 help="Do you consume alcohol heavily?")
            
            ment_hlth = st.slider("Mental Health Issues (days/month)", 
                                min_value=0, max_value=30, 
                                value=0, step=1,
                                help="Number of days in the past month with mental health issues")
            
            diff_walk = st.selectbox("Difficulty Walking", 
                                   options=[0, 1],
                                   format_func=lambda x: "No" if x == 0 else "Yes",
                                   help="Do you have serious difficulty walking or climbing stairs?")
        
        # Submit button
        submit_button = st.form_submit_button("Predict Diabetes Risk")
    
    # Process the form submission
    if submit_button:
        # Collect all features
        features = np.array([
            hb, cholesterol, bmi, heart_disease, phys_activity, phys_hlth,
            gender, age, stroke, gen_hlth, chol_check, smoker,
            fruits, veggies, alcohol, ment_hlth, diff_walk
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Display prediction result
        st.subheader("Prediction Result")
        
        # Create columns for result and visualization
        result_col, viz_col = st.columns([1, 1])
        
        with result_col:
            if prediction == 1:
                st.error("⚠️ **High Risk of Diabetes Detected**")
                st.write(f"Probability: {probability:.2%}")
                st.write("""
                **Recommendations:**
                - Consider consulting with a healthcare provider
                - Monitor your blood glucose levels regularly
                - Maintain a healthy diet and exercise routine
                - Review your lifestyle factors shown in the risk assessment
                """)
            else:
                st.success("✅ **Low Risk of Diabetes Detected**")
                st.write(f"Probability: {probability:.2%}")
                st.write("""
                **Recommendations:**
                - Continue maintaining your healthy lifestyle
                - Regular check-ups are still recommended
                - Stay active and maintain a balanced diet
                """)
        
        with viz_col:
            # Create a gauge chart for risk visualization
            fig, ax = plt.subplots(figsize=(4, 3))
            
            # Create a simple gauge using a pie chart
            colors = ['#3498db', '#e74c3c']
            explode = (0, 0.1)
            wedges, texts, autotexts = ax.pie(
                [1-probability, probability], 
                autopct='%1.1f%%',
                explode=explode,
                colors=colors,
                startangle=90,
                wedgeprops={'alpha': 0.7}
            )
            
            # Change text color to white for better visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
            
            # Add a circle at the center to make it look like a gauge
            circle = plt.Circle((0, 0), 0.7, fc='white')
            ax.add_artist(circle)
            
            # Add title and text
            ax.text(0, 0, f"{probability:.1%}", ha='center', va='center', fontsize=20)
            ax.set_title("Diabetes Risk", fontsize=14)
            
            st.pyplot(fig)
        
        # Feature importance section
        st.subheader("Key Factors Influencing Your Risk")
        
        # Get feature importances from the model
        feature_names = [
            "HB", "Cholesterol", "BMI", "Heart Disease", "Physical Activity", 
            "Physical Health", "Gender", "Age", "Stroke", "General Health", 
            "Cholesterol Check", "Smoker", "Fruits", "Vegetables", 
            "Alcohol Consumption", "Mental Health", "Difficulty Walking"
        ]
        
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[-8:]  # Show top 8 features
        
        # Create horizontal bar chart of feature importances
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color='#2ecc71')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Relative Importance')
        ax.set_title('Top Factors Influencing Diabetes Risk')
        
        st.pyplot(fig)
        
        # Add user-specific insights based on their inputs
        st.subheader("Personalized Insights")
        
        insights = []
        
        if bmi > 30:
            insights.append("Your BMI is in the obese range, which significantly increases diabetes risk.")
        elif bmi > 25:
            insights.append("Your BMI is in the overweight range, which can increase diabetes risk.")
        
        if age > 45:
            insights.append("Being over 45 is a risk factor for Type 2 diabetes.")
        
        if heart_disease == 1:
            insights.append("Having heart disease increases your risk for diabetes.")
        
        if phys_activity == 0:
            insights.append("Lack of physical activity is associated with higher diabetes risk.")
        
        if smoker == 1:
            insights.append("Smoking can increase insulin resistance and diabetes risk.")
        
        if fruits == 0 and veggies == 0:
            insights.append("Regular consumption of fruits and vegetables is associated with lower diabetes risk.")
        
        # Display insights
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.write("No specific risk factors identified in your inputs.")

# STEP 5: CREATE THE MODEL INFORMATION PAGE
# =======================================
def model_info_page():
    st.header("Model Information")
    
    st.write("""
    ### Random Forest Classifier
    
    This application uses a **Random Forest Classifier** to predict diabetes risk. Random Forests are a popular
    machine learning algorithm that create multiple decision trees and merge their predictions to achieve
    higher accuracy and prevent overfitting.
    
    ### Model Features
    
    The model uses the following features to make predictions:
    
    | Feature | Description |
    | ------- | ----------- |
    | HB | Hemoglobin level |
    | Cholesterol | Blood cholesterol level |
    | BMI | Body Mass Index |
    | Heart_Disease | Presence of heart disease (0=No, 1=Yes) |
    | PhysActivity | Regular physical activity (0=No, 1=Yes) |
    | PhysHlth | Days with physical health issues in past month |
    | Gender | Gender (0=Female, 1=Male) |
    | Age | Age in years |
    | Stroke | History of stroke (0=No, 1=Yes) |
    | GenHlth | General health rating (1=Excellent, 5=Poor) |
    | CholCheck | Cholesterol check in past 5 years (0=No, 1=Yes) |
    | Smoker | Smoking status (0=No, 1=Yes) |
    | Fruits | Regular fruit consumption (0=No, 1=Yes) |
    | Veggies | Regular vegetable consumption (0=No, 1=Yes) |
    | HvyAlcoholConsump | Heavy alcohol consumption (0=No, 1=Yes) |
    | MentHlth | Days with mental health issues in past month |
    | DiffWalk | Difficulty walking (0=No, 1=Yes) |
    """)
    
    # Load model to display performance metrics
    data = load_data()
    model = load_or_train_model(data)
    
    # Display model performance metrics
    st.subheader("Model Performance")
    
    # Create sample test data for demonstration
    X = data.drop('Diabetes_State', axis=1)
    y = data['Diabetes_State']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate and display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        ax.set_yticklabels(['No Diabetes', 'Diabetes'])
        st.pyplot(fig)
    
    with col2:
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.dataframe(df_report.style.format("{:.2f}"))
        
        # Accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Get feature names and importances
    feature_names = X.columns
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(range(len(importances)), importances[indices], align='center')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(feature_names[indices], rotation=90)
    ax.set_title('Feature Importance')
    ax.set_ylabel('Importance')
    st.pyplot(fig)

# STEP 6: CREATE THE DATA EXPLORATION PAGE
# ======================================
def data_exploration_page():
    st.header("Dataset Exploration")
    
    # Load data
    data = load_data()
    
    # Show dataset overview
    st.subheader("Dataset Overview")
    st.write(f"Dataset Shape: {data.shape}")
    st.dataframe(data.head())
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(data.describe())
    
    # Distribution of target variable
    st.subheader("Distribution of Diabetes Status")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    diabetes_counts = data['Diabetes_State'].value_counts()
    ax.pie(diabetes_counts, labels=['No Diabetes', 'Diabetes'], 
           autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
    ax.set_title('Distribution of Diabetes Status')
    st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    
    # Feature relationship analysis
    st.subheader("Feature Relationships")
    
    # Let user select features to explore
    feature_options = data.columns.tolist()
    feature_options.remove('Diabetes_State')  # Remove target variable from options
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("Select X-axis Feature", feature_options, index=2)  # BMI as default
    
    with col2:
        y_feature = st.selectbox("Select Y-axis Feature", feature_options, index=0)  # HB as default
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_feature, y=y_feature, hue='Diabetes_State', 
                   data=data, palette=['#3498db', '#e74c3c'], ax=ax)
    ax.set_title(f'{x_feature} vs {y_feature} by Diabetes Status')
    st.pyplot(fig)
    
    # Show histograms for selected features
    st.subheader("Feature Distributions by Diabetes Status")
    
    selected_feature = st.selectbox("Select Feature to Analyze", feature_options)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x=selected_feature, hue='Diabetes_State', 
                multiple='dodge', palette=['#3498db', '#e74c3c'], ax=ax)
    ax.set_title(f'Distribution of {selected_feature} by Diabetes Status')
    st.pyplot(fig)
    
    # Age group analysis
    st.subheader("Diabetes Prevalence by Age Group")
    
    # Create age groups
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 30, 45, 60, 75, 100], 
                            labels=['<30', '30-45', '46-60', '61-75', '>75'])
    
    # Group by age group and calculate diabetes prevalence
    age_diabetes = data.groupby('AgeGroup')['Diabetes_State'].mean().reset_index()
    age_diabetes['Prevalence'] = age_diabetes['Diabetes_State'] * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='AgeGroup', y='Prevalence', data=age_diabetes, ax=ax)
    ax.set_title('Diabetes Prevalence by Age Group')
    ax.set_ylabel('Diabetes Prevalence (%)')
    ax.set_xlabel('Age Group')
    st.pyplot(fig)
    
    # BMI category analysis
    st.subheader("Diabetes Prevalence by BMI Category")
    
    # Create BMI categories
    data['BMI_Category'] = pd.cut(data['BMI'], 
                                bins=[0, 18.5, 25, 30, 35, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])
    
    # Group by BMI category and calculate diabetes prevalence
    bmi_diabetes = data.groupby('BMI_Category')['Diabetes_State'].mean().reset_index()
    bmi_diabetes['Prevalence'] = bmi_diabetes['Diabetes_State'] * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='BMI_Category', y='Prevalence', data=bmi_diabetes, ax=ax)
    ax.set_title('Diabetes Prevalence by BMI Category')
    ax.set_ylabel('Diabetes Prevalence (%)')
    ax.set_xlabel('BMI Category')
    st.pyplot(fig)

# STEP 7: RUN THE APPLICATION
# ==========================
if __name__ == '__main__':
    main()