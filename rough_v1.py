import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Dict, List, Tuple, Union
from math import radians, sin, cos, sqrt, atan2
import warnings
import joblib
warnings.filterwarnings('ignore')

class EnhancedAccidentPredictorV5:
    """
    Enhanced accident prediction model with route assessment capabilities.
    Includes sophisticated risk scoring, route analysis, and weather/infrastructure interactions.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.feature_names = None
        self.categorical_mappings = {}
        self.default_category = "Unknown"
        
        # Initialize components
        self._initialize_features()
        self._initialize_preprocessors()
        self._initialize_model()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _initialize_features(self):
        """Initialize feature groups used in the model"""
        self.numeric_features = [
            'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)'
        ]
        self.categorical_features = ['State', 'Weather_Condition']
        self.binary_features = ['Crossing', 'Junction', 'Traffic_Signal']
        self.time_features = [
            'hour', 'day_of_week', 'month',
            'is_weekend', 'is_rush_hour', 'is_night'
        ]
        self.interaction_features = [
            'visibility_night', 'rush_hour_rain', 'temp_humidity',
            'morning_rush', 'evening_rush', 'weather_rush_hour',
            'temp_weather_risk'
        ]
        
    def _initialize_preprocessors(self):
        """Initialize data preprocessing components"""
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(
            strategy='constant',
            fill_value=self.default_category
        )
        self.scaler = StandardScaler()
        self.label_encoders = {
            feature: LabelEncoder() 
            for feature in self.categorical_features
        }
        
    def _initialize_model(self):
        """Initialize the RandomForest model with optimized parameters"""
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.7,
            random_state=42,
            n_jobs=-1
        )

    def haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate great circle distance between two points in miles"""
        R = 3959.87433  # Earth's radius in miles

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp data"""
        df_copy = df.copy()
        
        try:
            df_copy['Start_Time'] = pd.to_datetime(
                df_copy['Start_Time'],
                errors='coerce'
            )
            
            if df_copy['Start_Time'].isnull().any():
                current_time = pd.Timestamp.now()
                df_copy['Start_Time'].fillna(current_time, inplace=True)
            
            # Extract basic time features
            df_copy['hour'] = df_copy['Start_Time'].dt.hour
            df_copy['day_of_week'] = df_copy['Start_Time'].dt.dayofweek
            df_copy['month'] = df_copy['Start_Time'].dt.month
            
            # Create derived time features
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
            df_copy['is_rush_hour'] = (
                ((df_copy['hour'] >= 7) & (df_copy['hour'] <= 9)) |
                ((df_copy['hour'] >= 16) & (df_copy['hour'] <= 18))
            ).astype(int)
            df_copy['is_night'] = (
                (df_copy['hour'] >= 20) | (df_copy['hour'] <= 5)
            ).astype(int)
            
        except Exception as e:
            logging.error(f"Error in time feature creation: {str(e)}")
            for feature in self.time_features:
                df_copy[feature] = 0
                
        return df_copy

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables"""
        df_copy = df.copy()
        
        # Weather-time interactions
        df_copy['visibility_night'] = df_copy['Visibility(mi)'] * df_copy['is_night']
        df_copy['rush_hour_rain'] = (
            df_copy['is_rush_hour'] * 
            (df_copy['Weather_Condition'] == 'Rain').astype(int)
        )
        
        # Environmental interactions
        df_copy['temp_humidity'] = (
            df_copy['Temperature(F)'] * df_copy['Humidity(%)'] / 100
        )
        
        # Traffic timing features
        df_copy['morning_rush'] = (
            (df_copy['hour'] >= 7) & 
            (df_copy['hour'] <= 9)
        ).astype(int)
        df_copy['evening_rush'] = (
            (df_copy['hour'] >= 16) & 
            (df_copy['hour'] <= 18)
        ).astype(int)
        
        # Weather risk scores
        df_copy['weather_rush_hour'] = (
            df_copy['is_rush_hour'] * 
            df_copy['Weather_Condition'].map({
                'Clear': 0.3,
                'Rain': 0.7,
                'Snow': 0.9,
                'Fog': 0.8,
                'Thunderstorm': 1.0
            }).fillna(0.5)
        )
        
        # Temperature-weather interaction
        df_copy['temp_weather_risk'] = (
            abs(df_copy['Temperature(F)'] - 70) / 40 *
            df_copy['Weather_Condition'].map({
                'Clear': 0.5,
                'Rain': 0.8,
                'Snow': 1.0,
                'Fog': 0.7,
                'Thunderstorm': 0.9
            }).fillna(0.6)
        ).clip(0, 1)
        
        return df_copy

    def _preprocess_features(
        self,
        df: pd.DataFrame,
        train_mode: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess features for model training or prediction.
        Handles missing values, encoding, and feature creation.
        """
        try:
            df_processed = df.copy()
            
            # Create time features
            df_processed = self._create_time_features(df_processed)
            
            # Handle numeric features
            numeric_data = df_processed[self.numeric_features].copy()
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            
            if train_mode:
                numeric_data = pd.DataFrame(
                    self.numeric_imputer.fit_transform(numeric_data),
                    columns=self.numeric_features,
                    index=df_processed.index
                )
                numeric_data = pd.DataFrame(
                    self.scaler.fit_transform(numeric_data),
                    columns=self.numeric_features,
                    index=df_processed.index
                )
            else:
                numeric_data = pd.DataFrame(
                    self.numeric_imputer.transform(numeric_data),
                    columns=self.numeric_features,
                    index=df_processed.index
                )
                numeric_data = pd.DataFrame(
                    self.scaler.transform(numeric_data),
                    columns=self.numeric_features,
                    index=df_processed.index
                )
            
            for col in self.numeric_features:
                df_processed[col] = numeric_data[col]
            
            # Handle categorical features
            for feature in self.categorical_features:
                df_processed[feature] = df_processed[feature].astype(str)
                
                if train_mode:
                    unique_categories = set(df_processed[feature].unique())
                    unique_categories.add(self.default_category)
                    
                    self.label_encoders[feature].fit(list(unique_categories))
                    
                    df_processed[feature] = self.categorical_imputer.fit_transform(
                        df_processed[feature].values.reshape(-1, 1)
                    ).ravel()
                    
                    df_processed[feature] = self.label_encoders[feature].transform(
                        df_processed[feature]
                    )
                    
                    self.categorical_mappings[feature] = dict(
                        zip(
                            self.label_encoders[feature].classes_,
                            self.label_encoders[feature].transform(
                                self.label_encoders[feature].classes_
                            )
                        )
                    )
                else:
                    known_categories = set(self.label_encoders[feature].classes_)
                    df_processed[feature] = df_processed[feature].map(
                        lambda x: x if x in known_categories else self.default_category
                    )
                    
                    df_processed[feature] = self.categorical_imputer.transform(
                        df_processed[feature].values.reshape(-1, 1)
                    ).ravel()
                    
                    df_processed[feature] = self.label_encoders[feature].transform(
                        df_processed[feature]
                    )
            
            # Handle binary features
            for feature in self.binary_features:
                df_processed[feature] = pd.to_numeric(
                    df_processed[feature].fillna(0),
                    errors='coerce'
                ).fillna(0).astype(int)
            
            # Create interaction features
            df_processed = self._create_interaction_features(df_processed)
            
            # Combine all features
            feature_cols = (
                self.numeric_features +
                self.categorical_features +
                self.binary_features +
                self.time_features +
                self.interaction_features
            )
            
            # Update feature names if in training mode
            if train_mode:
                self.feature_names = feature_cols
            
            result = df_processed[feature_cols]
            
            # Final validation
            null_counts = result.isnull().sum()
            if null_counts.any():
                problematic_cols = null_counts[null_counts > 0].index.tolist()
                raise ValueError(f"NaN values found in columns: {problematic_cols}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def create_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create sophisticated risk score combining multiple risk factors.
        Returns normalized risk scores between 0 and 1.
        """
        try:
            # Base risk factors
            time_risk = pd.Series(0.3, index=df.index)
            
            # Time-based risk modifiers
            rush_multiplier = np.where(df['is_rush_hour'] == 1, 1.3, 1.0)
            night_multiplier = np.where(df['is_night'] == 1, 1.2, 1.0)
            
            # Weather and environmental risks
            visibility_risk = (1 - (df['Visibility(mi)'].fillna(10) / 10)).clip(0, 1) * 0.3
            wind_risk = (df['Wind_Speed(mph)'].fillna(0) / 40).clip(0, 1) * 0.2
            
            # Temperature risks
            temp = df['Temperature(F)'].fillna(70)
            cold_risk = np.where(temp < 32, (32 - temp) / 32, 0).clip(0, 1) * 0.2
            heat_risk = np.where(temp > 90, (temp - 90) / 30, 0).clip(0, 1) * 0.2
            temp_risk = np.maximum(cold_risk, heat_risk)
            
            # Infrastructure risk
            infrastructure_risk = (
                df['Crossing'].fillna(0) * 0.20 +
                df['Junction'].fillna(0) * 0.25 +
                df['Traffic_Signal'].fillna(0) * -0.15  # Signals reduce risk
            ).clip(0, 0.5)
            
            # Weather condition multipliers
            weather_multiplier = df['Weather_Condition'].map({
                'Clear': 1.0,
                'Rain': 1.4,
                'Snow': 1.6,
                'Fog': 1.5,
                'Thunderstorm': 1.7
            }).fillna(1.2)
            
            # Calculate final risk score
            base_risk = time_risk * rush_multiplier * night_multiplier * weather_multiplier
            additional_risks = (
                visibility_risk +
                wind_risk +
                temp_risk +
                infrastructure_risk
            )
            
            final_score = (base_risk + additional_risks).clip(0, 1)
            
            # Add small random noise to prevent perfect prediction
            noise = np.random.normal(0, 0.02, size=len(final_score))
            final_score = (final_score + noise).clip(0, 1)
            
            return final_score.values
            
        except Exception as e:
            logging.error(f"Error creating target variable: {str(e)}")
            raise

    def train(
        self,
        df: pd.DataFrame,
        cv_folds: int = 5
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Train model with cross-validation and return performance metrics.
        
        Parameters:
        df: DataFrame containing training data
        cv_folds: Number of cross-validation folds
        
        Returns:
        Dictionary containing:
            - cross_val_metrics: Dict of mean and std dev for MAE, MSE, R2
            - final_metrics: Dict of metrics on full training set
            - feature_importance: Dict of feature importances
        """
        try:
            logging.info("Starting model training with cross-validation")
            
            X = self._preprocess_features(df, train_mode=True)
            y = self.create_target(df)
            
            if X.isnull().any().any() or np.isnan(y).any():
                raise ValueError("NaN values found in features or target")
                
            # Initialize cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Track metrics across folds
            mae_scores = []
            mse_scores = []
            r2_scores = []
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model on this fold
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_val)
                
                # Calculate metrics
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                mae_scores.append(mae)
                mse_scores.append(mse)
                r2_scores.append(r2)
                
                logging.info(
                    f"Fold {fold}/{cv_folds} - "
                    f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}"
                )
            
            # Calculate cross-validation metrics
            cross_val_metrics = {
                'mae': {
                    'mean': np.mean(mae_scores),
                    'std': np.std(mae_scores)
                },
                'mse': {
                    'mean': np.mean(mse_scores),
                    'std': np.std(mse_scores)
                },
                'r2': {
                    'mean': np.mean(r2_scores),
                    'std': np.std(r2_scores)
                }
            }
            
            # Train final model on full dataset
            self.model.fit(X, y)
            final_predictions = self.model.predict(X)
            
            # Calculate final metrics
            final_metrics = {
                'mae': mean_absolute_error(y, final_predictions),
                'mse': mean_squared_error(y, final_predictions),
                'r2': r2_score(y, final_predictions)
            }
            
            # Calculate feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            # Sort features by importance
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Mark model as fitted
            self.is_fitted = True
            
            logging.info("Model training completed successfully")
            
            return {
                'cross_val_metrics': cross_val_metrics,
                'final_metrics': final_metrics,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        df: DataFrame containing features for prediction
        
        Returns:
        Array of risk scores between 0 and 1
        """
        try:
            if not self.is_fitted:
                raise ValueError(
                    "Model must be trained before making predictions. "
                    "Call train() first."
                )
            
            # Preprocess features
            X = self._preprocess_features(df, train_mode=False)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Ensure predictions are between 0 and 1
            predictions = np.clip(predictions, 0, 1)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

    def evaluate(
        self,
        df: pd.DataFrame,
        detailed: bool = False
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluate model performance on a test dataset.
        
        Parameters:
        df: DataFrame containing test data
        detailed: Whether to return additional metrics
        
        Returns:
        Dictionary of evaluation metrics
        """
        try:
            if not self.is_fitted:
                raise ValueError(
                    "Model must be trained before evaluation. "
                    "Call train() first."
                )
            
            # Create predictions
            X = self._preprocess_features(df, train_mode=False)
            y_true = self.create_target(df)
            y_pred = self.predict(df)
            
            # Calculate basic metrics
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
            
            if detailed:
                # Add detailed error analysis
                errors = np.abs(y_true - y_pred)
                metrics.update({
                    'max_error': np.max(errors),
                    'error_std': np.std(errors),
                    'error_percentiles': {
                        '25%': np.percentile(errors, 25),
                        '50%': np.percentile(errors, 50),
                        '75%': np.percentile(errors, 75),
                        '90%': np.percentile(errors, 90)
                    }
                })
                
                # Add feature-specific analysis
                high_error_mask = errors > np.percentile(errors, 75)
                problematic_features = {}
                
                for feature in self.feature_names:
                    if feature in df.columns:
                        feature_values = df[feature].values
                        correlation = np.corrcoef(feature_values, errors)[0, 1]
                        problematic_features[feature] = correlation
                
                metrics['feature_error_correlations'] = problematic_features
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """
        Save the trained model and its components to disk.
        
        Parameters:
        filepath: Path where model should be saved
        """
        try:
            if not self.is_fitted:
                raise ValueError("Cannot save untrained model")
                
            model_data = {
                'model': self.model,
                'numeric_imputer': self.numeric_imputer,
                'categorical_imputer': self.categorical_imputer,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'categorical_mappings': self.categorical_mappings,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(model_data, filepath)
            logging.info(f"Model successfully saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedAccidentPredictorV5':
        """
        Load a trained model from disk.
        
        Parameters:
        filepath: Path to saved model
        
        Returns:
        Loaded model instance
        """
        try:
            # Create new instance
            instance = cls()
            
            # Load saved components
            model_data = joblib.load(filepath)
            
            # Restore all components
            instance.model = model_data['model']
            instance.numeric_imputer = model_data['numeric_imputer']
            instance.categorical_imputer = model_data['categorical_imputer']
            instance.scaler = model_data['scaler']
            instance.label_encoders = model_data['label_encoders']
            instance.categorical_mappings = model_data['categorical_mappings']
            instance.feature_names = model_data['feature_names']
            instance.is_fitted = model_data['is_fitted']
            
            logging.info(f"Model successfully loaded from {filepath}")
            return instance
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

def main():
    """
    Example usage of the accident prediction model.
    Demonstrates training, evaluation, and prediction workflow.
    """
    # Create example training data
    train_data = pd.DataFrame({
        'Start_Time': pd.date_range('2024-01-01', '2024-01-10', freq='H'),
        'Temperature(F)': np.random.normal(65, 15, 240),
        'Humidity(%)': np.random.normal(70, 10, 240),
        'Pressure(in)': np.random.normal(30, 1, 240),
        'Visibility(mi)': np.random.normal(9, 2, 240),
        'Wind_Speed(mph)': np.random.uniform(0, 20, 240),
        'Distance(mi)': np.random.uniform(0.1, 2, 240),
        'State': np.random.choice(['CA', 'NY', 'TX', 'FL'], 240),
        'Weather_Condition': np.random.choice(
            ['Clear', 'Rain', 'Snow', 'Fog'],
            240,
            p=[0.7, 0.15, 0.1, 0.05]
        ),
        'Crossing': np.random.choice([0, 1], 240, p=[0.8, 0.2]),
        'Junction': np.random.choice([0, 1], 240, p=[0.7, 0.3]),
        'Traffic_Signal': np.random.choice([0, 1], 240, p=[0.75, 0.25])
    })
    
    # Create test data
    test_data = pd.DataFrame({
        'Start_Time': pd.date_range('2024-01-11', '2024-01-12', freq='H'),
        'Temperature(F)': np.random.normal(65, 15, 24),
        'Humidity(%)': np.random.normal(70, 10, 24),
        'Pressure(in)': np.random.normal(30, 1, 24),
        'Visibility(mi)': np.random.normal(9, 2, 24),
        'Wind_Speed(mph)': np.random.uniform(0, 20, 24),
        'Distance(mi)': np.random.uniform(0.1, 2, 24),
        'State': np.random.choice(['CA', 'NY', 'TX', 'FL'], 24),
        'Weather_Condition': np.random.choice(
            ['Clear', 'Rain', 'Snow', 'Fog'],
            24,
            p=[0.7, 0.15, 0.1, 0.05]
        ),
        'Crossing': np.random.choice([0, 1], 24, p=[0.8, 0.2]),
        'Junction': np.random.choice([0, 1], 24, p=[0.7, 0.3]),
        'Traffic_Signal': np.random.choice([0, 1], 24, p=[0.75, 0.25])
    })
    
    # Create and train model
    model = EnhancedAccidentPredictorV5()
    
    print("Training model...")
    training_results = model.train(train_data)
    
    print("\nCross-validation results:")
    for metric, values in training_results['cross_val_metrics'].items():
        print(f"{metric.upper()}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  Std:  {values['std']:.4f}")
    
    print("\nEvaluating on test data...")
    test_metrics = model.evaluate(test_data, detailed=True)
    
    print("\nTest set metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, dict):
            print(f"\n{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # Make predictions on new data
    example_data = pd.DataFrame({
        'Start_Time': [pd.Timestamp.now()],
        'Temperature(F)': [72.0],
        'Humidity(%)': [65.0],
        'Pressure(in)': [30.1],
        'Visibility(mi)': [10.0],
        'Wind_Speed(mph)': [8.5],
        'Distance(mi)': [0.5],
        'State': ['CA'],
        'Weather_Condition': ['Clear'],
        'Crossing': [0],
        'Junction': [1],
        'Traffic_Signal': [1]
    })
    
    print("\nMaking prediction on example data...")
    prediction = model.predict(example_data)[0]
    print(f"Predicted risk score: {prediction:.4f}")
    
    # Save and reload model
    print("\nSaving model...")
    model.save_model('accident_predictor_v5.joblib')
    
    print("Loading saved model...")
    loaded_model = EnhancedAccidentPredictorV5.load_model('accident_predictor_v5.joblib')
    
    # Verify loaded model works
    loaded_prediction = loaded_model.predict(example_data)[0]
    print(f"Prediction from loaded model: {loaded_prediction:.4f}")

if __name__ == "__main__":
    main()