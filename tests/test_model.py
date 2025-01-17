import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.utils.monitoring import MLTestSuite

@pytest.fixture
def sample_classification_data():
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=2, 
        random_state=42
    )
    return X, y

def test_data_integrity(sample_classification_data):
    X, y = sample_classification_data
    df = pd.DataFrame(X)
    
    # Test data integrity
    MLTestSuite.test_data_integrity(
        df, 
        expected_columns=list(range(X.shape[1])),
        min_rows=500
    )

def test_model_performance(sample_classification_data):
    X, y = sample_classification_data
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Test performance
    performance = MLTestSuite.test_model_performance(
        model, 
        X_test, 
        y_test,
        min_accuracy=0.7,
        min_precision=0.6,
        min_recall=0.6
    )
    
    # Additional assertions if needed
    assert performance['accuracy'] > 0.7
