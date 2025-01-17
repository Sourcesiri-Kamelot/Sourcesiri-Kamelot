import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, raw_data_path):
        self.raw_data = pd.read_csv(raw_data_path)
        self.preprocessed_data = None
    
    def clean_data(self):
        # Remove duplicates
        self.raw_data.drop_duplicates(inplace=True)
        
        # Handle missing values
        self.raw_data.fillna(self.raw_data.mean(), inplace=True)
        
        return self
    
    def encode_categorical(self, categorical_columns):
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            self.raw_data[col] = le.fit_transform(self.raw_data[col])
            label_encoders[col] = le
        
        return label_encoders
    
    def scale_features(self, feature_columns):
        # Scale numerical features
        scaler = StandardScaler()
        self.raw_data[feature_columns] = scaler.fit_transform(self.raw_data[feature_columns])
        
        return scaler
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        # Separate features and target
        X = self.raw_data.drop(columns=[target_column])
        y = self.raw_data[target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
