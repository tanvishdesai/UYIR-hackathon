from grpc import FutureCancelledError
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import MiniBatchKMeans
import logging
from typing import Dict, List, Tuple, Union, Optional
import warnings
from datetime import datetime
import shap
import dask
from distributed import TimeoutError
from dask import dataframe as dd
from dask.distributed import Client
import joblib

warnings.filterwarnings('ignore')

class EnhancedAccidentPredictorV5:
    """Enhanced accident prediction model optimized for large datasets"""
    

    def __init__(self, batch_size: int = 50000):
        # Initialize logging
        self._setup_advanced_logging()
        
        # Initialize state
        self.is_fitted = False
        self.feature_names = None
        self.categorical_mappings = {}
        self.default_category = "Unknown"
        self.batch_size = batch_size
        
        # Initialize components
        self._initialize_features()
        self._initialize_preprocessors()
        self._initialize_model()

    def _setup_advanced_logging(self):
        """Set up detailed logging for model operations"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_filename = f"accident_model_{datetime.now():%Y%m%d_%H%M}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _get_serializable_state(self, predictor):
        """Extract only the necessary components needed for preprocessing"""
        return {
            'numeric_features': predictor.numeric_features,
            'categorical_features': predictor.categorical_features,
            'binary_features': predictor.binary_features,
            'time_features': predictor.time_features,
            'geo_features': predictor.geo_features,
            'interaction_features': predictor.interaction_features,
            'metadata_features': predictor.metadata_features,
            'default_category': predictor.default_category,
            'categorical_mappings': predictor.categorical_mappings,
            # Include fitted preprocessors state
            'numeric_imputer_statistics_': predictor.numeric_imputer.statistics_ if hasattr(predictor.numeric_imputer, 'statistics_') else None,
            'scaler_mean_': predictor.scaler.mean_ if hasattr(predictor.scaler, 'mean_') else None,
            'scaler_scale_': predictor.scaler.scale_ if hasattr(predictor.scaler, 'scale_') else None,
            'label_encoder_classes_': {
                feature: encoder.classes_ 
                for feature, encoder in predictor.label_encoders.items()
                if hasattr(encoder, 'classes_')
            }
        }

    def _initialize_features(self):
        """Initialize feature groups and advanced feature definitions"""
        # Original features
        self.numeric_features = [
            'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)'
        ]
        
        self.categorical_features = ['State', 'Weather_Condition']
        self.binary_features = ['Crossing', 'Junction', 'Traffic_Signal']
        
        self.time_features = [
            'hour', 'day_of_week', 'month',
            'is_weekend', 'is_rush_hour', 'is_night',
            'hour_sin', 'hour_cos'
        ]
        
        self.geo_features = [] # We'll populate this based on available data
        self.required_geo_columns = ['Start_Lat', 'Start_Lng']
          
        self.interaction_features = [
            'visibility_night', 'rush_hour_rain', 'temp_humidity',
            'morning_rush', 'evening_rush', 'weather_rush_hour',
            'temp_weather_risk', 'historical_hour_risk',
            'historical_dow_risk', 'historical_month_risk'
        ]
        
        # Add metadata columns that should be preserved
        self.metadata_features = ['Start_Time']
   
    def _initialize_preprocessors(self):
        """Initialize data preprocessing components"""
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(
            strategy='constant',
            fill_value=self.default_category
        )
        self.scaler = StandardScaler()
        self.label_encoders = {
            feature: LabelEncoder() for feature in self.categorical_features
        }  
        
        self.geo_clusterer = MiniBatchKMeans(
            n_clusters=100,
            batch_size=10000,
            random_state=42
        )
    
    def _initialize_model(self):
        """Initialize the primary prediction model"""
        self.model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
            validation_fraction=0.1,
            early_stopping=True,
            n_iter_no_change=5
        )
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced temporal features including cyclical encodings"""
        df_copy = df.copy()
        
        try:
            df_copy['Start_Time'] = pd.to_datetime(
                df_copy['Start_Time'],
                errors='coerce'
            )
            
            if df_copy['Start_Time'].isnull().any():
                current_time = pd.Timestamp.now()
                df_copy['Start_Time'].fillna(current_time, inplace=True)
            
            # Basic time features
            df_copy['hour'] = df_copy['Start_Time'].dt.hour
            df_copy['day_of_week'] = df_copy['Start_Time'].dt.dayofweek
            df_copy['month'] = df_copy['Start_Time'].dt.month
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
            
            # Rush hour features
            df_copy['is_rush_hour'] = (
                ((df_copy['hour'] >= 7) & (df_copy['hour'] <= 9)) |
                ((df_copy['hour'] >= 16) & (df_copy['hour'] <= 18))
            ).astype(int)
            
            df_copy['is_night'] = (
                (df_copy['hour'] >= 20) | (df_copy['hour'] <= 5)
            ).astype(int)
            
            # Cyclical time encodings
            hour_rad = 2 * np.pi * df_copy['hour'] / 24
            df_copy['hour_sin'] = np.sin(hour_rad)
            df_copy['hour_cos'] = np.cos(hour_rad)
            
        except Exception as e:
            self.logger.error(f"Error in time feature creation: {str(e)}")
            for feature in self.time_features:
                df_copy[feature] = 0
        
        return df_copy
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical accident patterns"""
        df_copy = df.copy()
        
        try:
            timestamp = pd.to_datetime(df_copy['Start_Time'])
            
            # Calculate historical risk patterns
            df_copy['historical_hour_risk'] = (
                df_copy.groupby(timestamp.dt.hour)['Start_Time']
                .transform('count') / len(df_copy)
            )
            
            df_copy['historical_dow_risk'] = (
                df_copy.groupby(timestamp.dt.dayofweek)['Start_Time']
                .transform('count') / len(df_copy)
            )
            
            df_copy['historical_month_risk'] = (
                df_copy.groupby(timestamp.dt.month)['Start_Time']
                .transform('count') / len(df_copy)
            )
            
        except Exception as e:
            self.logger.error(f"Error in historical feature creation: {str(e)}")
            df_copy['historical_hour_risk'] = 0
            df_copy['historical_dow_risk'] = 0
            df_copy['historical_month_risk'] = 0
        
        return df_copy    

    def _create_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographical clustering features with graceful fallback for missing data"""
        df_copy = df.copy()
        
        try:
            # Check if required geo columns are present
            has_geo_data = all(col in df_copy.columns for col in self.required_geo_columns)
            
            if has_geo_data:
                # Verify data quality
                valid_coords = (
                    df_copy[self.required_geo_columns]
                    .notna()
                    .all(axis=1)
                )
                
                if valid_coords.any():
                    # Use only valid coordinates for clustering
                    valid_data = df_copy[valid_coords]
                    coords = valid_data[self.required_geo_columns].values
                    
                    # Perform clustering on valid coordinates
                    cluster_labels = self.geo_clusterer.fit_predict(coords)
                    
                    # Initialize all clusters as -1 (invalid/missing)
                    df_copy['location_cluster'] = -1
                    
                    # Update only the valid rows with their cluster labels
                    df_copy.loc[valid_coords, 'location_cluster'] = cluster_labels
                    
                    # Add to geo features if not already present
                    if 'location_cluster' not in self.geo_features:
                        self.geo_features.append('location_cluster')
                    
                    self.logger.info(
                        f"Created geo clusters for {valid_coords.sum()} out of {len(df_copy)} records"
                    )
                else:
                    self.logger.warning("No valid coordinate pairs found in the dataset")
                    df_copy['location_cluster'] = -1
            else:
                self.logger.warning(
                    f"Missing required geo columns: {self.required_geo_columns}. "
                    "Proceeding without geographical features."
                )
                df_copy['location_cluster'] = -1
                
            # Create alternative location-based features if possible
            if 'State' in df_copy.columns:
                # Calculate state-level risk scores based on historical data
                state_risks = df_copy.groupby('State').size() / len(df_copy)
                df_copy['state_risk_score'] = df_copy['State'].map(state_risks)
                
                if 'state_risk_score' not in self.geo_features:
                    self.geo_features.append('state_risk_score')
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error in geo feature creation: {str(e)}")
            # Return DataFrame with dummy geo features
            df_copy['location_cluster'] = -1
            df_copy['state_risk_score'] = 0.5
            return df_copy      
   
    def _preprocess_features(self, df: pd.DataFrame, train_mode: bool = True) -> pd.DataFrame:
        """
        Complete feature preprocessing pipeline with proper datetime handling.
        This version ensures that temporal data is correctly processed into numerical features
        while preventing raw datetime values from entering the feature matrix.
        """
        try:
            df_processed = df.copy()
            
            # First, process temporal features
            # This needs to happen before we remove the Start_Time column
            if 'Start_Time' in df_processed.columns:
                # Convert to datetime if not already
                df_processed['Start_Time'] = pd.to_datetime(
                    df_processed['Start_Time'],
                    errors='coerce'
                )
                
                # Create time-based features before removing the datetime column
                df_processed = self._create_time_features(df_processed)
                df_processed = self._create_historical_features(df_processed)
                
                # Now we can safely drop the original datetime column
                # df_processed = df_processed.drop('Start_Time', axis=1)
            else:
                self.logger.warning("Start_Time column not found. Creating default temporal features.")
                # Create default values for time features
                for feature in self.time_features:
                    df_processed[feature] = 0
                
                for feature in ['historical_hour_risk', 'historical_dow_risk', 'historical_month_risk']:
                    df_processed[feature] = 0.5  # Default risk value
            
            # Create geographical features
            df_processed = self._create_geo_features(df_processed)
            
            # Process numeric features
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
            
            # Process categorical features
            for feature in self.categorical_features:
                if feature not in df_processed.columns:
                    self.logger.warning(f"Categorical feature {feature} not found. Using default category.")
                    df_processed[feature] = self.default_category
                
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
            
            # Process binary features
            for feature in self.binary_features:
                if feature not in df_processed.columns:
                    self.logger.warning(f"Binary feature {feature} not found. Using default value 0.")
                    df_processed[feature] = 0
                else:
                    df_processed[feature] = pd.to_numeric(
                        df_processed[feature].fillna(0),
                        errors='coerce'
                    ).fillna(0).astype(int)
            
            # Create interaction features
            df_processed['visibility_night'] = (
                df_processed['Visibility(mi)'] * df_processed['is_night']
            )
            df_processed['rush_hour_rain'] = (
                df_processed['is_rush_hour'] * 
                (df_processed['Weather_Condition'] == 
                self.categorical_mappings.get('Weather_Condition', {}).get('Rain', 0)
                ).astype(int)
            )
            df_processed['temp_humidity'] = (
                df_processed['Temperature(F)'] * df_processed['Humidity(%)']
            )
            
            # Select only the features we want to use for modeling
            feature_cols = (
                self.numeric_features +
                self.categorical_features +
                self.binary_features +
                self.time_features +
                self.geo_features +
                self.interaction_features
            )
            
            if train_mode:
                # Update feature names only during training
                self.feature_names = [f for f in feature_cols if f in df_processed.columns]
            
            # Select only the features that exist in our processed dataset
            result = df_processed[self.feature_names].copy()
            
            # Final validation
            final_null_counts = result.isnull().sum()
            if final_null_counts.any():
                problematic_columns = final_null_counts[final_null_counts > 0].index.tolist()
                self.logger.error(f"NaN values found in columns: {problematic_columns}")
                raise ValueError(f"NaN values found in columns: {problematic_columns}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    # [Rest of the class methods remain the same]  
  
     
    def train_with_validation(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ) -> List[Dict[str, Dict[str, float]]]:
        """Train with time-based validation"""
        try:
            self.logger.info("Starting time-based validation")

            # Ensure Start_Time is present
            if 'Start_Time' not in df.columns:
                raise ValueError("Start_Time column is required for time-based validation")
            
            df['timestamp'] = pd.to_datetime(df['Start_Time'])
            df = df.sort_values('timestamp')
            
            split_points = np.linspace(0, len(df), n_splits + 1, dtype=int)
            metrics_history = []
            
            for i in range(n_splits - 1):
                train_df = df.iloc[split_points[i]:split_points[i + 1]]
                test_df = df.iloc[split_points[i + 1]:split_points[i + 2]]
                
                # Process training data in batches
                self.logger.info(f"Processing training split {i + 1}/{n_splits - 1}")
                train_metrics = self.train(train_df)
                
                # Process test data in batches
                self.logger.info(f"Processing test split {i + 1}/{n_splits - 1}")
                test_metrics = self.evaluate(test_df)
                
                metrics_history.append({
                    'train': train_metrics,
                    'test': test_metrics
                })
                
                self.logger.info(f"Completed validation split {i + 1}/{n_splits - 1}")
            
            return metrics_history
               
        except Exception as e:
            self.logger.error(f"Error in time-based validation: {str(e)}")
            raise
    

 
    def create_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create sophisticated risk score incorporating multiple factors.
        Modified to handle missing columns more gracefully.
        """
        try:
            # Base risk factors
            time_risk = pd.Series(0.3, index=df.index)
            
            # Time-based risk (with null checks)
            rush_multiplier = np.where(
                df.get('is_rush_hour', pd.Series(0, index=df.index)) == 1,
                1.3, 1.0
            )
            night_multiplier = np.where(
                df.get('is_night', pd.Series(0, index=df.index)) == 1,
                1.2, 1.0
            )
            
            # Weather and environmental risks
            visibility_risk = (1 - (df.get('Visibility(mi)', pd.Series(10, index=df.index)) / 10)).clip(0, 1) * 0.3
            wind_risk = (df.get('Wind_Speed(mph)', pd.Series(0, index=df.index)) / 40).clip(0, 1) * 0.2
            
            # Temperature extremes risk
            temp = df.get('Temperature(F)', pd.Series(70, index=df.index))
            cold_risk = np.where(temp < 32, (32 - temp) / 32, 0).clip(0, 1) * 0.2
            heat_risk = np.where(temp > 90, (temp - 90) / 30, 0).clip(0, 1) * 0.2
            temp_risk = np.maximum(cold_risk, heat_risk)
            
            # Infrastructure risk
            infrastructure_risk = (
                df.get('Crossing', pd.Series(0, index=df.index)) * 0.20 +
                df.get('Junction', pd.Series(0, index=df.index)) * 0.25 +
                df.get('Traffic_Signal', pd.Series(0, index=df.index)) * -0.15
            ).clip(0, 0.5)
            
            # Historical pattern risk (with null checks)
            historical_risk = (
                df.get('historical_hour_risk', pd.Series(0.5, index=df.index)) * 0.4 +
                df.get('historical_dow_risk', pd.Series(0.5, index=df.index)) * 0.3 +
                df.get('historical_month_risk', pd.Series(0.5, index=df.index)) * 0.3
            )
            
            # Weather condition specific risks (with safer mapping)
            default_weather_multiplier = 1.2
            if 'Weather_Condition' in df.columns and hasattr(self, 'categorical_mappings'):
                weather_mapping = {
                    self.categorical_mappings.get('Weather_Condition', {}).get('Clear', -1): 1.0,
                    self.categorical_mappings.get('Weather_Condition', {}).get('Rain', -1): 1.4,
                    self.categorical_mappings.get('Weather_Condition', {}).get('Snow', -1): 1.6,
                    self.categorical_mappings.get('Weather_Condition', {}).get('Fog', -1): 1.5,
                    self.categorical_mappings.get('Weather_Condition', {}).get('Thunderstorm', -1): 1.7
                }
                weather_multiplier = df['Weather_Condition'].map(
                    lambda x: weather_mapping.get(x, default_weather_multiplier)
                )
            else:
                weather_multiplier = pd.Series(default_weather_multiplier, index=df.index)
            
            # Calculate final risk score
            base_risk = time_risk * rush_multiplier * night_multiplier * weather_multiplier
            additional_risks = (
                visibility_risk +
                wind_risk +
                temp_risk +
                infrastructure_risk +
                historical_risk * 0.2  # Weight historical patterns
            )
            
            final_score = (base_risk + additional_risks).clip(0, 1)
            
            # Add small random noise to prevent perfect prediction
            noise = np.random.normal(0, 0.02, size=len(final_score))
            final_score = (final_score + noise).clip(0, 1)
            
            return final_score.values
            
        except Exception as e:
            self.logger.error(f"Error creating target variable: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions with uncertainty handling"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
            
        try:
            X = self._preprocess_features(df, train_mode=False)
            predictions = self.model.predict(X)
            
            if predictions.max() == predictions.min():
                # Fallback predictions based on features
                visibility_score = 1 - (df['Visibility(mi)'].fillna(10) / 10).clip(0, 1)
                weather_score = df['Weather_Condition'].map({
                    'Clear': 0.3,
                    'Rain': 0.6,
                    'Snow': 0.8,
                    'Fog': 0.7,
                    'Thunderstorm': 0.9
                }).fillna(0.5)
                
                return (visibility_score * 0.6 + weather_score * 0.4).clip(0, 1)
            
            # Scale predictions while preserving relative differences
            scaled_predictions = (
                predictions - predictions.min()
            ) / (predictions.max() - predictions.min())
            
            # Apply sigmoid transformation
            return 1 / (1 + np.exp(-6 * (scaled_predictions - 0.5)))
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return np.full(len(df), 0.5)
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
            
        try:
            X = self._preprocess_features(df, train_mode=False)
            y_true = self.create_target(df)
            y_pred = self.model.predict(X)
            
            metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'prediction_mean': np.mean(y_pred),
                'prediction_std': np.std(y_pred)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {str(e)}")
            raise
    
    def explain_prediction(
        self,
        df: pd.DataFrame,
        sample_size: int = None
    ) -> pd.DataFrame:
        """Generate SHAP values for model interpretability"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before generating explanations")
            
        try:
            X = self._preprocess_features(df, train_mode=False)
            
            if sample_size and len(X) > sample_size:
                X = X.sample(n=sample_size, random_state=42)
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            return pd.DataFrame(
                shap_values,
                columns=self.feature_names,
                index=X.index
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP values: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """Save model and all preprocessing components"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
            
        try:
            model_data = {
                'model': self.model,
                'numeric_imputer': self.numeric_imputer,
                'categorical_imputer': self.categorical_imputer,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'categorical_mappings': self.categorical_mappings,
                'is_fitted': self.is_fitted
            }
            
            if self.use_distributed:
                model_data['geo_clusterer'] = self.geo_clusterer
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnhancedAccidentPredictorV5':
            """Load saved model and restore state"""
            try:
                instance = cls()
                
                model_data = joblib.load(filepath)
                
                instance.model = model_data['model']
                instance.numeric_imputer = model_data['numeric_imputer']
                instance.categorical_imputer = model_data['categorical_imputer']
                instance.scaler = model_data['scaler']
                instance.label_encoders = model_data['label_encoders']
                instance.feature_names = model_data['feature_names']
                instance.categorical_mappings = model_data['categorical_mappings']
                instance.is_fitted = model_data['is_fitted']
                
                if 'geo_clusterer' in model_data:
                    instance.geo_clusterer = model_data['geo_clusterer']
                    instance.use_distributed = True
                
                instance.logger.info(f"Model loaded successfully from {filepath}")
                return instance
                
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Extract feature importances from the trained model.
        HistGradientBoostingRegressor uses 'model.permutation_importance_' 
        rather than 'feature_importances_'.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importances")
            
        try:
            # For HistGradientBoostingRegressor, we need to calculate permutation importance
            from sklearn.inspection import permutation_importance
            
            # Get the most recently processed features
            if not hasattr(self, '_last_processed_features'):
                self.logger.warning("No processed features available for importance calculation")
                return {}
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model,
                self._last_processed_features,
                self._last_processed_target,
                n_repeats=5,
                random_state=42
            )
            
            # Create importance dictionary using mean importance scores
            importance_dict = dict(zip(
                self.feature_names,
                result.importances_mean
            ))
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importances: {str(e)}")
            return {}

    def train(self, df: pd.DataFrame) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Train the model using batch processing to handle large datasets efficiently.
        Modified to properly handle feature importance calculation.
        """
        try:
            self.logger.info(f"Starting model training with batch size {self.batch_size}")
            
            # Initialize lists to store processed data
            X_processed = []
            y_processed = []
            
            # Process data in batches
            for start_idx in range(0, len(df), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                # First preprocess features
                X_batch = self._preprocess_features(batch, train_mode=True)
                
                # Create target using the preprocessed batch
                preprocessed_batch = batch.copy()
                for col in X_batch.columns:
                    preprocessed_batch[col] = X_batch[col]
                
                y_batch = self.create_target(preprocessed_batch)
                
                X_processed.append(X_batch)
                y_processed.append(y_batch)
                
                self.logger.info(f"Processed batch {start_idx//self.batch_size + 1}")
            
            # Combine processed batches
            X = pd.concat(X_processed, axis=0)
            y = np.concatenate(y_processed)
            
            # Store last processed data for feature importance calculation
            self._last_processed_features = X
            self._last_processed_target = y
            
            # Verify data integrity before training
            if X.isnull().any().any():
                problematic_columns = X.columns[X.isnull().any()].tolist()
                raise ValueError(f"NaN values found in features: {problematic_columns}")
                
            if np.isnan(y).any():
                raise ValueError("NaN values found in target variable")
            
            # Train the model with early stopping
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Calculate predictions and metrics
            y_pred = self.model.predict(X)
            
            metrics = {
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'prediction_mean': np.mean(y_pred),
                'prediction_std': np.std(y_pred)
            }
            
            # Calculate feature importances using the new method
            metrics['feature_importances'] = self.get_feature_importances()
            
            self.logger.info("Model training completed successfully")
            return metrics
           
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

def main():
    """Example usage with large dataset using batch processing"""
    try:
        predictor = EnhancedAccidentPredictorV5(batch_size=50000)
        filepath = '/kaggle/input/mini-us-accident/downsized.csv'
        
        print("Loading data in batches...")
        # First, just load the data without preprocessing
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=50000):
            chunks.append(chunk)
          
        # Combine all processed chunks
        processed_data = pd.concat(chunks, axis=0)
        print(f"Total records processed: {len(processed_data)}")
        
        # Split data into training and test sets
        print("\nSplitting data into training and test sets...")
        train_data, test_data = train_test_split(
            processed_data,
            test_size=0.2,
            random_state=42
        )
        print(f"Training set size: {len(train_data)}")
        print(f"Test set size: {len(test_data)}")
        
        # Train with time-based validation
        print("\nStarting time-based validation...")
        validation_metrics = predictor.train_with_validation(
            train_data,
            n_splits=5
        )
        
        # Print validation results with clear formatting
        print("\n=== Validation Results ===")
        for i, metrics in enumerate(validation_metrics):
            print(f"\nValidation Split {i + 1}:")
            print("Training Metrics:")
            for metric, value in metrics['train'].items():
                if metric != 'feature_importances':
                    print(f"  {metric}: {value:.4f}")
                else:
                    print("\nFeature Importances:")
                    sorted_importances = sorted(
                        value.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for feature, importance in sorted_importances[:10]:
                        print(f"  {feature}: {importance:.4f}")
            
            print("\nTest Metrics:")
            for metric, value in metrics['test'].items():
                if metric != 'feature_importances':
                    print(f"  {metric}: {value:.4f}")
        
        # Final evaluation on test set
        print("\n=== Final Test Set Evaluation ===")
        final_metrics = predictor.evaluate(test_data)
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save the model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_filename = f'accident_predictor_v6_{timestamp}.joblib'
        predictor.save_model(model_filename)
        print(f"\nModel saved successfully as {model_filename}!")
        
        return predictor, validation_metrics, final_metrics
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    predictor, validation_metrics, final_metrics = main()