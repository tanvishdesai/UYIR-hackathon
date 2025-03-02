from typing import Dict, List
import pandas as pd
import numpy as np
import pytz
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import holidays
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedAccidentPredictorV2:
    def __init__(self, n_splits: int = 5):
        """Initialize the enhanced accident prediction model with advanced components"""
        # Initialize imputers with KNN for better accuracy
        self.numeric_imputer = KNNImputer(
            n_neighbors=5,
            weights='distance'
        )
        self.label_encoders = {}
        self.target_encoders = {}
        self.scaler = StandardScaler()
        
        # Initialize model (parameters will be tuned during training)
        self.model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
        
        # Define feature groups
        self.numeric_features = [
            'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
            'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)'
        ]
        
        # Columns to drop due to high missing values
        self.columns_to_drop = [
            'End_Lat', 'End_Lng', 'Wind_Chill(F)',
            'Precipitation(in)', 'Wind_Direction'
        ]
        
        self.high_cardinality_features = ['City', 'County']
        self.low_cardinality_features = ['State', 'Weather_Condition']
        
        self.binary_features = [
            'Crossing', 'Junction', 'Traffic_Signal', 
            'is_night', 'is_weekend', 'is_rush_hour', 
            'is_holiday'
        ]
        self.kfold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        # Initialize US holidays
        self.us_holidays = holidays.US()
              # Store preprocessing states
        self.fitted_preprocessors = False
        self.fitted_model = False
        self.timezone = pytz.timezone('US/Eastern')

        # Track important features
        self.important_features = None
        self.feature_importance_threshold = 0.01  # Keep features with importance > 1%
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            float: MAPE value
        """
        try:
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except Exception as e:
            logger.error(f"Error calculating MAPE: {str(e)}")
            return None
    
    def calculate_poisson_loss(self, y_true, y_pred):
        """Calculate Poisson loss for count data
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            float: Poisson loss value
        """
        try:
            return np.mean(y_pred - y_true * np.log(y_pred))
        except Exception as e:
            logger.error(f"Error calculating Poisson loss: {str(e)}")
            return None
    

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and encode enhanced time-based features with timezone handling
        
        Args:
            df: Input dataframe with timestamp column
            
        Returns:
            DataFrame with additional time-based features
        """
        logger.info("Creating time-based features")
        df_copy = df.copy()
        
        try:
            # Convert Start_Time to datetime with timezone awareness
            df_copy['Start_Time'] = pd.to_datetime(df_copy['Start_Time'])
            df_copy['Start_Time'] = df_copy['Start_Time'].dt.tz_localize(
                'UTC'
            ).dt.tz_convert(self.timezone)
            
            # Extract time components with timezone consideration
            local_time = df_copy['Start_Time']
            df_copy['hour'] = local_time.dt.hour
            df_copy['day_of_week'] = local_time.dt.dayofweek
            df_copy['month'] = local_time.dt.month
            df_copy['year'] = local_time.dt.year
            df_copy['day_of_year'] = local_time.dt.dayofyear
            
            # Create temporal indicators
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
            
            # Define rush hours considering local time
            df_copy['is_rush_hour'] = (
                ((df_copy['hour'] >= 7) & (df_copy['hour'] <= 9)) |
                ((df_copy['hour'] >= 16) & (df_copy['hour'] <= 18))
            ).astype(int)
            
            # Check holidays with proper date conversion
            df_copy['is_holiday'] = df_copy['Start_Time'].dt.date.map(
                lambda x: x in self.us_holidays
            ).astype(int)
            
            # Night time consideration
            df_copy['is_night'] = (
                (df_copy['hour'] >= 20) | (df_copy['hour'] <= 5)
            ).astype(int)
            
            # Add seasonal indicators
            df_copy['season'] = pd.cut(
                df_copy['day_of_year'],
                bins=[0, 80, 172, 264, 366],
                labels=['Winter', 'Spring', 'Summer', 'Fall']
            )
            
            logger.info("Time features created successfully")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error creating time features: {str(e)}")
            raise

    def _perform_cross_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_metrics: List[str] = None
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation with multiple metrics
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_metrics: List of metric names to evaluate
            
        Returns:
            Dictionary of cross-validation scores
        """
        if cv_metrics is None:
            cv_metrics = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
        
        try:
            logger.info(f"Performing {self.kfold.n_splits}-fold cross-validation")
            
            # Define scoring metrics
            scoring = {
                'neg_mse': 'neg_mean_squared_error',
                'r2': 'r2',
                'neg_mae': 'neg_mean_absolute_error'
            }
            
            # Perform cross-validation with multiple metrics
            cv_results = cross_validate(
                self.model,
                X,
                y,
                cv=self.kfold,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Process and format results
            cv_scores = {
                'test_rmse': np.sqrt(-cv_results['test_neg_mse']),
                'train_rmse': np.sqrt(-cv_results['train_neg_mse']),
                'test_r2': cv_results['test_r2'],
                'train_r2': cv_results['train_r2'],
                'test_mae': -cv_results['test_neg_mae'],
                'train_mae': -cv_results['train_neg_mae']
            }
            
            # Log cross-validation results
            logger.info("\nCross-validation results:")
            for metric, scores in cv_scores.items():
                logger.info(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            return cv_scores
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model with cross-validation and comprehensive evaluation
        
        Args:
            df: Training data
            
        Returns:
            Dictionary containing model performance metrics
        """
        logger.info("Starting model training process with cross-validation")
        
        try:
            # Preprocess features
            X = self.preprocess_data(df, train_mode=True)
            y = self.create_target_variable(df)
            
            # Perform cross-validation before final training
            cv_scores = self._perform_cross_validation(X, y)
            
            # Handle imbalanced data if needed
            if len(np.unique(y)) < 10:
                logger.info("Applying SMOTE for imbalanced data")
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
            
            # Tune hyperparameters
            self._tune_hyperparameters(X, y)
            
            # Train final model
            self.model.fit(X, y)
            self.fitted_model = True
            
            # Select important features
            self.important_features = self._select_important_features(X)
            
            # Make predictions for final metrics
            predictions = self.model.predict(X)
            
            # Calculate comprehensive metrics
            metrics = {
                'r2_score': r2_score(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'explained_variance': np.var(predictions) / np.var(y),
                'cv_rmse_mean': cv_scores['test_rmse'].mean(),
                'cv_rmse_std': cv_scores['test_rmse'].std(),
                'cv_r2_mean': cv_scores['test_r2'].mean(),
                'cv_r2_std': cv_scores['test_r2'].std()
            }
            
            logger.info("Model training completed successfully")
            logger.info("\nFinal Model Performance Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def _handle_missing_values(self, df, train_mode=True):
        """Handle missing values with advanced strategies
        
        Args:
            df (pd.DataFrame): Input dataframe
            train_mode (bool): Whether in training or prediction mode
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        try:
            df_copy = df.copy()
            
            # Drop columns with high missing values
            df_copy = df_copy.drop(self.columns_to_drop, axis=1, errors='ignore')
            
            # Use KNN imputation for remaining numeric features
            if train_mode:
                df_copy[self.numeric_features] = self.numeric_imputer.fit_transform(
                    df_copy[self.numeric_features]
                )
            else:
                df_copy[self.numeric_features] = self.numeric_imputer.transform(
                    df_copy[self.numeric_features]
                )
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def _handle_imbalanced_data(self, X, y):
        """Handle imbalanced data using SMOTE
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable
            
        Returns:
            tuple: Balanced X and y arrays
        """
        try:
            logger.info("Applying SMOTE to balance the dataset")
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Error applying SMOTE: {str(e)}")
            logger.warning("Proceeding with original imbalanced data")
            return X, y
    
    def _tune_hyperparameters(self, X, y):
        """Tune model hyperparameters using RandomizedSearchCV
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable
        """
        try:
            logger.info("Starting hyperparameter tuning")
            
            # Define parameter space
            param_dist = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt'],
            }
            
            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                self.model,
                param_distributions=param_dist,
                n_iter=20,
                cv=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit RandomizedSearchCV
            random_search.fit(X, y)
            
            # Update model with best parameters
            self.model = random_search.best_estimator_
            logger.info(f"Best parameters found: {random_search.best_params_}")
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            logger.warning("Using default hyperparameters")
    
    def _select_important_features(self, X, feature_names):
        """Select important features based on feature importance scores
        
        Args:
            X (pd.DataFrame): Feature matrix
            feature_names (list): List of feature names
            
        Returns:
            list: Names of important features
        """
        try:
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create dictionary of feature importances
            feature_importance = dict(zip(feature_names, importances))
            
            # Select features above threshold
            self.important_features = [
                feature for feature, importance in feature_importance.items()
                if importance > self.feature_importance_threshold
            ]
            
            logger.info(f"Selected {len(self.important_features)} important features")
            return self.important_features
            
        except Exception as e:
            logger.error(f"Error selecting important features: {str(e)}")
            return feature_names
    
    
    def predict_risk_score(self, new_data):
        """Predict accident risk scores with error handling
        
        Args:
            new_data (pd.DataFrame): New data for prediction
            
        Returns:
            np.array: Predicted risk scores (0-1 scale)
        """
        try:
            # Handle missing values
            new_data_processed = self._handle_missing_values(new_data, train_mode=False)
            
            # Preprocess the new data
            X_new = self.preprocess_data(new_data_processed, train_mode=False)
            
            # Use only important features if available
            if self.important_features:
                X_new = X_new[self.important_features]
            
            # Make predictions
            predictions = self.model.predict(X_new)
            
            # Convert to risk scores (0-1 scale)
            risk_scores = (predictions - predictions.min()) / (
                predictions.max() - predictions.min()
            )
            
            return risk_scores
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('your_accident_data.csv')
        logger.info("Data loaded successfully")
        
        # Initialize and train model
        predictor = EnhancedAccidentPredictorV2()
        metrics = predictor.train(df)
        
        # Example prediction
        new_data = pd.DataFrame({
            'Start_Time': ['2023-01-01 08:00:00'],
            'Temperature(F)': [70],
            'Humidity(%)': [80],
            'Pressure(in)': [29.92],
            'Visibility(mi)': [10],
            'Wind_Speed(mph)': [5],
            'City': ['Dayton'],
            'County': ['Montgomery'],
            'State': ['OH'],
            'Weather_Condition': ['Clear'],
            'Crossing': [False],
            'Junction': [True],
            'Traffic_Signal': [True],
            'Distance(mi)': [0.5]
        })
        
        risk_scores = predictor.predict_risk_score(new_data)
        logger.info(f"Predicted risk scores: {risk_scores}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")