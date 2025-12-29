import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def train_and_save_model(data_path):
    # 1. Load Data (Now using read_csv for your loan_data.csv)
    try:
        df = pd.read_csv(data_path)
        print(f"--- Step 1: Loaded {data_path} successfully ---")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Feature Engineering (Module 1.4)
    # Creating the ratio we found important during EDA
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    df['int_burden'] = (df['loan_int_rate'] / 100) * df['loan_amnt']
    
    # 3. Separate Features and Target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # 4. Data Preparation (Module 1.2)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 5. Optimized Model Building (Module 3.3 & 4.3)
    # Using the RBF kernel and C=10 found during tuning
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            kernel='rbf', 
            C=10, 
            gamma='scale', 
            class_weight='balanced', 
            probability=True
        ))
    ])
    
    # 6. Final Training
    print("--- Step 2: Training final RBF SVM model... ---")
    model_pipeline.fit(X, y)
    
    # 7. Model Deployment Preparation (Module 4.1)
    joblib.dump(model_pipeline, 'credit_model.pkl')
    print("--- Step 3: SUCCESS! Model saved as 'credit_model.pkl' ---")

if __name__ == "__main__":
    # Ensure this matches your actual filename
    train_and_save_model('loan_data.csv')