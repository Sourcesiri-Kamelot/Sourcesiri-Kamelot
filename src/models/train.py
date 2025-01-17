import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from src.data.preprocessing import DataPreprocessor
from src.utils.monitoring import MLModelMonitor

def train_model(data_path, target_column, feature_columns, categorical_columns):
    # Initialize monitor
    monitor = MLModelMonitor('ml_training')
    
    # Preprocess data
    preprocessor = DataPreprocessor(data_path)
    preprocessor.clean_data()
    preprocessor.encode_categorical(categorical_columns)
    preprocessor.scale_features(feature_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(target_column)
    
    # MLflow tracking
    mlflow.set_experiment('model_training')
    
    with mlflow.start_run():
        # Initialize and train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_params({
            'n_estimators': 100,
            'random_state': 42
        })
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics({
            'train_accuracy': train_score,
            'test_accuracy': test_score
        })
        
        # Log the model
        mlflow.sklearn.log_model(model, 'model')
        
        # Log monitoring metrics
        monitor.log_system_metrics()
        
        return model, test_score

if __name__ == '__main__':
    # Example usage
    train_model(
        data_path='data/raw_data.csv',
        target_column='target',
        feature_columns=['feature1', 'feature2', 'feature3'],
        categorical_columns=['category']
    )
