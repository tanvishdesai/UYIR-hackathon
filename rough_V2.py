import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timezone
import pytz
from math import radians, sin, cos, sqrt, atan2
import logging
import os
import json
import joblib
from tqdm.auto import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RouteSegment:
    """Data class to store route segment information"""
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    distance: float
    road_type: str
    speed_limit: Optional[float] = None
    traffic_signals: int = 0
    intersections: int = 0

class AccidentRiskPredictor:
    """
    Enhanced accident risk prediction model designed specifically for US accident data.
    Includes route analysis capabilities and advanced feature engineering.
    """
    
    def __init__(self, timezone_default: str = 'US/Eastern', model_dir: str = 'saved_models'):
        # Add model directory parameter
        self.model_dir = model_dir
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
     
     
        self.is_fitted = False
        self.feature_names = None
        self.categorical_mappings = {}
        self.timezone_default = timezone_default
        self.default_category = "Unknown"
        
        # Initialize components
        self._initialize_features()
        self._initialize_preprocessors()
        self._initialize_models()
        
        # Setup logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        self.logger = logging.getLogger('AccidentRiskPredictor')
        

    def save_model(self, prefix: str = '') -> str:
        """
        Save the trained model and all necessary components for later use.
        Handles NumPy data type serialization.
        
        Parameters:
        prefix: Optional string to prefix the model directory name
        
        Returns:
        str: Path to the saved model directory
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
            
        try:
            # Create timestamp for unique directory name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = os.path.join(
                self.model_dir,
                f"{prefix}model_{timestamp}" if prefix else f"model_{timestamp}"
            )
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the random forest and gradient boosting models
            joblib.dump(self.rf_model, os.path.join(model_dir, 'rf_model.joblib'))
            joblib.dump(self.gb_model, os.path.join(model_dir, 'gb_model.joblib'))
            
            # Save preprocessing components
            preprocessing_components = {
                'scaler': self.scaler,
                'weather_imputer': self.weather_imputer,
                'categorical_imputer': self.categorical_imputer,
                'label_encoders': self.label_encoders
            }
            joblib.dump(
                preprocessing_components,
                os.path.join(model_dir, 'preprocessing.joblib')
            )
            
            # Convert categorical mappings to JSON serializable format
            serializable_mappings = {}
            for feature, mapping in self.categorical_mappings.items():
                serializable_mappings[feature] = {
                    str(k): int(v) if isinstance(v, np.integer) else v
                    for k, v in mapping.items()
                }
            
            # Create serializable metadata
            model_metadata = {
                'feature_names': list(self.feature_names),  # Convert to list if it's numpy array
                'categorical_mappings': serializable_mappings,
                'weather_features': list(self.weather_features),
                'infrastructure_features': list(self.infrastructure_features),
                'categorical_features': list(self.categorical_features),
                'time_features': list(self.time_features),
                'location_features': list(self.location_features),
                'weather_severity': {
                    str(k): float(v) if isinstance(v, np.floating) else v
                    for k, v in self.weather_severity.items()
                },
                'timezone_default': str(self.timezone_default),
                'default_category': str(self.default_category)
            }
            
            # Save metadata as JSON
            with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                json.dump(model_metadata, f, indent=4)
                
            self.logger.info(f"Model saved successfully to: {model_dir}")
            return model_dir
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    @classmethod
    def load_model(cls, model_dir: str) -> 'AccidentRiskPredictor':
        """
        Load a previously saved model and all its components.
        
        Parameters:
        model_dir: Directory containing the saved model files
        
        Returns:
        AccidentRiskPredictor: Loaded model instance
        """
        try:
            # Load metadata first to initialize the model
            with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                
            # Create new instance with loaded timezone
            instance = cls(
                timezone_default=metadata['timezone_default'],
                model_dir=os.path.dirname(model_dir)
            )
            
            # Load preprocessing components
            preprocessing = joblib.load(
                os.path.join(model_dir, 'preprocessing.joblib')
            )
            instance.scaler = preprocessing['scaler']
            instance.weather_imputer = preprocessing['weather_imputer']
            instance.categorical_imputer = preprocessing['categorical_imputer']
            instance.label_encoders = preprocessing['label_encoders']
            
            # Load models
            instance.rf_model = joblib.load(
                os.path.join(model_dir, 'rf_model.joblib')
            )
            instance.gb_model = joblib.load(
                os.path.join(model_dir, 'gb_model.joblib')
            )
            
            # Set metadata attributes
            instance.feature_names = metadata['feature_names']
            instance.categorical_mappings = metadata['categorical_mappings']
            instance.weather_features = metadata['weather_features']
            instance.infrastructure_features = metadata['infrastructure_features']
            instance.categorical_features = metadata['categorical_features']
            instance.time_features = metadata['time_features']
            instance.location_features = metadata['location_features']
            instance.weather_severity = metadata['weather_severity']
            instance.default_category = metadata['default_category']
            
            # Mark as fitted
            instance.is_fitted = True
            
            instance.logger.info(f"Model loaded successfully from: {model_dir}")
            return instance
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
 
 
    def _initialize_features(self):
        """
        Initialize feature groups based on US accident dataset structure.
        Groups features by their nature and importance for prediction.
        """
        # Core environmental features
        self.weather_features = [
            'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
            'Precipitation(in)'
        ]
        
        # Road infrastructure features
        self.infrastructure_features = [
            'Crossing', 'Junction', 'Traffic_Signal', 'Stop',
            'Railway', 'Roundabout', 'Station', 'Amenity',
            'Bump', 'Give_Way', 'No_Exit', 'Traffic_Calming',
            'Turning_Loop'
        ]

        
           # Categorical features requiring encoding
        self.categorical_features = [
            'State', 'Weather_Condition', 'Wind_Direction',
            'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
            'Astronomical_Twilight'
        ]
        
        # Time-based features to be engineered
        self.time_features = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'is_rush_hour', 'is_holiday', 'season'
        ]
        
        # Location-based features
        self.location_features = ['Start_Lat', 'Start_Lng']
        
        # Severity feature (target variable alternative)
        self.severity_feature = ['Severity']        
   
    def _initialize_preprocessors(self):
        """Initialize data preprocessing components with specific strategies"""
        # Different imputation strategies for different feature types
        self.weather_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(
            strategy='constant',
            fill_value=self.default_category
        )
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        
        # Initialize encoders for categorical variables
        self.label_encoders = {
            feature: LabelEncoder() 
            for feature in self.categorical_features
        }
        
        # Special handling for weather conditions
        self.weather_severity = {
            'Clear': 0.1,
            'Cloudy': 0.2,
            'Overcast': 0.3,
            'Rain': 0.6,
            'Heavy Rain': 0.8,
            'Snow': 0.7,
            'Heavy Snow': 0.9,
            'Fog': 0.7,
            'Haze': 0.4,
            'Thunderstorm': 0.9
        }
        
    def _initialize_models(self):
        """
        Initialize ensemble of models for better prediction accuracy.
        Uses both RandomForest and GradientBoosting for their complementary strengths.
        """
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )

    def _process_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time-based features from timestamp data.
        Handles timezone conversion and special time periods.
        """
        df_processed = df.copy()
        
        try:
            # Convert timestamps to datetime
            df_processed['Start_Time'] = pd.to_datetime(
                df_processed['Start_Time'],
                errors='coerce'
            )
            
            # Handle missing timestamps
            if df_processed['Start_Time'].isnull().any():
                print("Missing timestamps detected, using current time")
                current_time = pd.Timestamp.now()
                df_processed['Start_Time'].fillna(current_time, inplace=True)
            
            local_tz = pytz.timezone(self.timezone_default)


            if df_processed['Start_Time'].dt.tz is not None:
            # If already tz-aware, convert to the desired timezone
                df_processed['Start_Time'] = df_processed['Start_Time'].dt.tz_convert(local_tz)
            else:
            # If tz-naive, localize to UTC first, then convert to the desired timezone
                df_processed['Start_Time'] = df_processed['Start_Time'].dt.tz_localize('UTC').dt.tz_convert(local_tz)
            
            # Extract basic time features
            df_processed['hour'] = df_processed['Start_Time'].dt.hour
            df_processed['day_of_week'] = df_processed['Start_Time'].dt.dayofweek
            df_processed['month'] = df_processed['Start_Time'].dt.month
            df_processed['season'] = (df_processed['month'] % 12 + 3) // 3
            
            # Create derived time features
            df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6])
            df_processed['is_rush_hour'] = (
                ((df_processed['hour'] >= 7) & (df_processed['hour'] <= 9)) |
                ((df_processed['hour'] >= 16) & (df_processed['hour'] <= 18))
            )
            
            # US holidays (simplified version - extend as needed)
            holidays = [
                # Format: (month, day)
                (1, 1),   # New Year's Day
                (7, 4),   # Independence Day
                (12, 25), # Christmas
                # Add more holidays as needed
            ]
            
            df_processed['is_holiday'] = df_processed.apply(
                lambda row: (row['month'], row['Start_Time'].day) in holidays,
                axis=1
            )
            
            return df_processed
            
        except Exception as e:
            print(f"Error in time feature processing: {str(e)}")
            raise
            
    def _process_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process location-based features and create derived geographical features.
        """
        df_processed = df.copy()
        
        try:
            # Calculate basic location features
            lat_bounds = (24.396308, 49.384358)  # Approximate US boundaries
            lon_bounds = (-125.000000, -66.934570)
            
            # Normalize coordinates to [0,1] range
            df_processed['norm_lat'] = (
                (df_processed['Start_Lat'] - lat_bounds[0]) /
                (lat_bounds[1] - lat_bounds[0])
            ).clip(0, 1)
            
            df_processed['norm_lng'] = (
                (df_processed['Start_Lng'] - lon_bounds[0]) /
                (lon_bounds[1] - lon_bounds[0])
            ).clip(0, 1)
            
            return df_processed
            
        except Exception as e:
            print(f"Error in location feature processing: {str(e)}")
            raise

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables, with proper handling of
        missing values and boolean conversions.
        
        Parameters:
        df: DataFrame containing the accident data
        
        Returns:
        DataFrame with added interaction features
        """
        df_processed = df.copy()
        
        try:
            # First, ensure all infrastructure features are properly converted to boolean
            # and handle missing values
            infrastructure_columns = ['Traffic_Signal', 'Junction', 'Crossing', 'Stop']
            for col in infrastructure_columns:
                if col in df_processed.columns:
                    # Convert string 'TRUE'/'FALSE' to boolean
                    df_processed[col] = df_processed[col].map({'TRUE': True, 'FALSE': False})
                    # Fill NaN with False
                    df_processed[col] = df_processed[col].fillna(False)
                    # Now safely convert to int (True -> 1, False -> 0)
                    df_processed[col] = df_processed[col].astype(int)

            # Weather-time interactions (ensure 'visibility_risk' exists)
            if 'visibility_risk' in df_processed.columns:
                df_processed['night_visibility_risk'] = (
                    df_processed['visibility_risk'] *
                    (df_processed['Sunrise_Sunset'] == 'Night').astype(int)
                )
            else:
                df_processed['night_visibility_risk'] = 0.0

            # Calculate traffic density proxy with safe integer conversion
            traffic_density_proxy = (
                df_processed['Traffic_Signal'] * 0.3 +
                df_processed['Junction'] * 0.3 +
                df_processed['Crossing'] * 0.2 +
                df_processed['Stop'] * 0.2
            )
            
            # Rush hour interactions using proxy
            if 'is_rush_hour' in df_processed.columns:
                df_processed['rush_hour_congestion'] = (
                    df_processed['is_rush_hour'].astype(int) *
                    (1 + traffic_density_proxy)
                )
            
            # Infrastructure-weather interactions
            if 'weather_severity' in df_processed.columns:
                df_processed['signal_weather_risk'] = (
                    df_processed['Traffic_Signal'] *
                    df_processed['weather_severity']
                )
            
            # Create twilight risk factor with proper boolean conversion
            df_processed['twilight_risk'] = (
                ((df_processed['Civil_Twilight'] == 'Night') |
                (df_processed['Nautical_Twilight'] == 'Night')).astype(int) * 0.5
            )
            
            # Weather-infrastructure combined risk
            weather_severity = df_processed.get('weather_severity', 
                                            pd.Series([0.5] * len(df_processed)))
            df_processed['combined_risk'] = (
                weather_severity *
                (1 + traffic_density_proxy) *
                (1 + df_processed['twilight_risk'])
            )
            
            return df_processed
            
        except Exception as e:
            raise ValueError(f"Error creating interaction features: {str(e)}")
  
    def _process_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and engineer weather-related features from the US accident dataset.
        Handles the specific weather condition formats and creates derived features.
        
        Parameters:
        df: DataFrame containing the accident data
        
        Returns:
        DataFrame with processed weather features
        """
        df_processed = df.copy()
        
        try:
            # Handle missing weather values
            weather_data = df_processed[self.weather_features].copy()
            weather_data = weather_data.replace([np.inf, -np.inf], np.nan)
            
            # Impute missing values
            weather_data = pd.DataFrame(
                self.weather_imputer.fit_transform(weather_data),
                columns=self.weather_features,
                index=df_processed.index
            )
            
            # Map weather conditions to severity scores
            weather_mapping = {
                'Clear': 0.1,
                'Fair': 0.2,
                'Cloudy': 0.3,
                'Overcast': 0.4,
                'Light Rain': 0.5,
                'Rain': 0.6,
                'Heavy Rain': 0.8,
                'Light Snow': 0.6,
                'Snow': 0.7,
                'Heavy Snow': 0.9,
                'Fog': 0.7,
                'Haze': 0.4,
                'Thunderstorm': 0.9
            }
            
            # Create weather severity score with default for unknown conditions
            df_processed['weather_severity'] = df_processed['Weather_Condition'].map(
                weather_mapping
            ).fillna(0.5)  # Default value for unknown conditions
            
            # Calculate visibility risk (inverse relationship)
            df_processed['visibility_risk'] = (
                1 - (df_processed['Visibility(mi)'].fillna(10) / 10)
            ).clip(0, 1)
            
            # Create precipitation risk when precipitation data is available
            if 'Precipitation(in)' in df_processed.columns:
                df_processed['precipitation_risk'] = (
                    df_processed['Precipitation(in)']
                    .fillna(0)
                    .clip(0, 1)
                )
            else:
                df_processed['precipitation_risk'] = 0.0
            
            # Temperature-based risk (extreme temperatures)
            temp_data = df_processed['Temperature(F)'].fillna(
                df_processed['Temperature(F)'].mean()
            )
            df_processed['temperature_risk'] = (
                (abs(temp_data - 70) / 40)  # Deviation from 70°F, normalized
            ).clip(0, 1)
            
            return df_processed
            
        except Exception as e:
            print(f"Error in weather feature processing: {str(e)}")
            raise
  
    def preprocess_data(
        self,
        df: pd.DataFrame,
        train_mode: bool = True
    ) -> pd.DataFrame:
        """
        Main preprocessing pipeline that combines all feature processing steps.
        
        Parameters:
        df: Input DataFrame
        train_mode: Whether preprocessing is for training or prediction
        
        Returns:
        Preprocessed DataFrame ready for model training or prediction
        """
        try:
            df_processed = df.copy()
                    # Convert boolean strings to actual booleans
            for col in self.infrastructure_features:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].map({'TRUE': True, 'FALSE': False})

            # Process each feature group
            df_processed = self._process_time_features(df_processed)
            df_processed = self._process_weather_features(df_processed)
            df_processed = self._process_location_features(df_processed)
            df_processed = self._create_interaction_features(df_processed)
            df_processed = self._handle_missing_values(df_processed)

            # Combine all features
            feature_cols = (
                self.weather_features +
                self.infrastructure_features +
                ['norm_lat', 'norm_lng'] +
                list(self.categorical_mappings.keys()) +
                self.time_features
            )
            
            if train_mode:
                self.feature_names = feature_cols
            
            return df_processed[feature_cols]
            
        except Exception as e:
            print(f"Error in main preprocessing pipeline: {str(e)}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values specific to your dataset format.
        """
        df_cleaned = df.copy()
        
        # Weather features
        for feature in self.weather_features:
            if feature in df_cleaned.columns:
                df_cleaned[feature] = pd.to_numeric(
                    df_cleaned[feature],
                    errors='coerce'
                )
                df_cleaned[feature].fillna(
                    df_cleaned[feature].median(),
                    inplace=True
                )
        
        # Infrastructure features
        for feature in self.infrastructure_features:
            if feature in df_cleaned.columns:
                df_cleaned[feature].fillna(False, inplace=True)
        
        # Categorical features
        for feature in self.categorical_features:
            if feature in df_cleaned.columns:
                df_cleaned[feature].fillna(self.default_category, inplace=True)
        
        return df_cleaned
    
    def calculate_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate comprehensive risk score combining multiple risk factors.
        This updated version ensures all required features are available before calculation.
        
        Parameters:
        df: Input DataFrame that has already been preprocessed
        
        Returns:
        Array of risk scores between 0 and 1
        """
        try:
            # First ensure we have all required features by preprocessing if needed
            if 'weather_severity' not in df.columns:
                df = self._process_weather_features(df)
            
            if 'visibility_risk' not in df.columns:
                # Add visibility risk calculation here if not already present
                df['visibility_risk'] = (
                    1 - (df['Visibility(mi)'].fillna(10) / 10)
                ).clip(0, 1)
            
            # Calculate precipitation-temperature risk if not present
            if 'precipitation_temp_risk' not in df.columns:
                temp_risk = (
                    abs(df['Temperature(F)'].fillna(70) - 70) / 40
                ).clip(0, 1)
                
                precip_risk = (
                    df['Precipitation(in)'].fillna(0).clip(0, 1)
                )
                
                df['precipitation_temp_risk'] = (
                    (temp_risk + precip_risk) / 2
                )

            # Now calculate the combined risk components
            # 1. Base environmental risk
            weather_risk = (
                df['weather_severity'] * 
                (1 + df['visibility_risk']) *
                (1 + df['precipitation_temp_risk'])
            )
            
            # 2. Infrastructure risk - handle missing values safely
            infrastructure_features = [
                'Traffic_Signal', 'Junction', 'Crossing', 'Stop'
            ]
            
            infrastructure_risk = pd.Series(0.0, index=df.index)
            for feature in infrastructure_features:
                if feature in df.columns:
                    # Convert to boolean then float, handling any missing values
                    feature_value = (
                        df[feature]
                        .map({'TRUE': True, 'FALSE': False})
                        .fillna(False)
                        .astype(float)
                    )
                    infrastructure_risk += feature_value * 0.3
            
            # 3. Time-based risk
            time_risk = pd.Series(0.0, index=df.index)
            
            # Add rush hour component if available
            if 'is_rush_hour' in df.columns:
                time_risk += df['is_rush_hour'].astype(float) * 0.4
                
            # Add holiday component if available
            if 'is_holiday' in df.columns:
                time_risk += df['is_holiday'].astype(float) * 0.3
                
            # Add night driving component
            time_risk += (
                (df['Sunrise_Sunset'] == 'Night').astype(float) * 0.3
            )
            
            # 4. Severity normalization (if available)
            if 'Severity' in df.columns:
                severity_normalized = (df['Severity'] - 1) / 3
            else:
                severity_normalized = pd.Series(0.5, index=df.index)
            
            # Combine all components with weights
            final_risk = (
                0.4 * severity_normalized +
                0.25 * weather_risk +
                0.2 * infrastructure_risk +
                0.15 * time_risk
            )
            
            # Ensure final risk is within [0,1] range
            return final_risk.clip(0, 1)
            
        except Exception as e:
            raise ValueError(f"Error calculating risk score: {str(e)}\n"
                            f"Available columns: {df.columns.tolist()}")
    
    def train(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Train the model using both RandomForest and GradientBoosting.
        
        Parameters:
        df: Training data
        cv_folds: Number of cross-validation folds
        
        Returns:
        Dictionary containing model performance metrics
        """
        try:
            print("Starting model training")
            training_results = {}
            
            # First process all features
            with tqdm(total=6, desc="Training Progress", unit="step") as pbar:
                processed_df = df.copy()
                
                # Process categorical features and store mappings
                for feature in tqdm(
                    self.categorical_features,
                    desc="Processing categorical features",
                    leave=False
                ):
                    if feature in processed_df.columns:
                        # Initialize or get existing encoder
                        encoder = self.label_encoders.get(feature, LabelEncoder())
                        
                        # Fill missing values before encoding
                        processed_df[feature].fillna(self.default_category, inplace=True)
                        
                        # Fit and transform
                        processed_df[feature] = encoder.fit_transform(processed_df[feature])
                        
                        # Store encoder and mapping
                        self.label_encoders[feature] = encoder
                        self.categorical_mappings[feature] = dict(
                            zip(encoder.classes_, encoder.transform(encoder.classes_))
                        )
                pbar.update(1)
                    
                # Process features with progress tracking
                pbar.set_description("Processing weather features")
                processed_df = self._process_weather_features(processed_df)
                pbar.update(1)
                
                pbar.set_description("Processing time features")
                processed_df = self._process_time_features(processed_df)
                pbar.update(1)
                
                pbar.set_description("Processing location features")
                processed_df = self._process_location_features(processed_df)
                pbar.update(1)
                
                pbar.set_description("Creating interaction features")
                processed_df = self._create_interaction_features(processed_df)
                processed_df = self._handle_missing_values(processed_df)
                
                # Calculate risk scores
                y = self.calculate_risk_score(processed_df)
                pbar.update(1)
        
                # Define and store feature columns
                self.feature_names = (
                    self.weather_features +
                    self.infrastructure_features +
                    ['norm_lat', 'norm_lng'] +
                    list(self.categorical_mappings.keys()) +
                    self.time_features
                )
                
                # Ensure all required features exist
                missing_features = [col for col in self.feature_names if col not in processed_df.columns]
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")
                
                # Get final feature matrix
                X = processed_df[self.feature_names]
                
                # Scale numerical features
                numerical_features = [
                    col for col in X.columns 
                    if X[col].dtype in ['float64', 'int64']
                ]
                X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
                
                # Log feature information
                print(f"Training with {len(self.feature_names)} features: {self.feature_names}")
                
                # Perform cross-validation with detailed metrics
                metrics = {
                    'rf': self._cross_validate(self.rf_model, X, y, cv_folds),
                    'gb': self._cross_validate(self.gb_model, X, y, cv_folds)
                }
                
                # Train final models
                print("\nTraining final models:")
                with tqdm(total=2, desc="Final model training", leave=False) as model_pbar:
                    self.rf_model.fit(X, y)
                    model_pbar.update(1)
                    self.gb_model.fit(X, y)
                    model_pbar.update(1)
                
                self.is_fitted = True
                pbar.update(1)
                
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance()
                
                # Prepare training results
                training_results = {
                    'data_statistics': {
                        'n_samples': len(df),
                        'n_features': len(self.feature_names)
                    },
                    'final_metrics': {
                        'rf_mae': metrics['rf']['mae']['mean'],
                        'rf_mse': metrics['rf']['mse']['mean'],
                        'rf_r2': metrics['rf']['r2']['mean'],
                        'gb_mae': metrics['gb']['mae']['mean'],
                        'gb_mse': metrics['gb']['mse']['mean'],
                        'gb_r2': metrics['gb']['r2']['mean']
                    },
                    'cross_val_metrics': metrics,
                    'feature_importance': feature_importance
                }
                
                # Save the trained model
                model_dir = self.save_model()
                training_results['model_saved_to'] = model_dir
                
                return training_results
                
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise
 
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate and combine feature importance from both models.
        Updated with additional error checking.
        
        Returns:
        Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained first")
            
        if self.feature_names is None:
            raise ValueError("Feature names not set. Train the model first.")
            
        # Get feature importance from both models
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        # Verify dimensions match
        if len(rf_importance) != len(self.feature_names):
            raise ValueError(
                f"Feature importance length ({len(rf_importance)}) "
                f"doesn't match feature names ({len(self.feature_names)})"
            )
        
        # Combine with weights (0.6 RF, 0.4 GB)
        combined_importance = 0.6 * rf_importance + 0.4 * gb_importance
        
        # Create feature importance dictionary with additional verification
        importance_dict = {
            name: importance 
            for name, importance in zip(self.feature_names, combined_importance)
        }
        
        # Sort by importance
        return dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))   
   
    def _cross_validate(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation and calculate performance metrics.
        
        Parameters:
        model: Model to evaluate
        X: Feature matrix
        y: Target values
        cv_folds: Number of cross-validation folds
        
        Returns:
        Dictionary of performance metrics
        """
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        metrics = {
            'mae': [],
            'mse': [],
            'r2': []
        }
        
        # Create progress bar for cross-validation
        fold_iterator = tqdm(
            enumerate(kf.split(X), 1),
            total=cv_folds,
            desc="Cross-validation",
            unit="fold"
        )



        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            fold_mae = mean_absolute_error(y_val, y_pred)
            fold_mse = mean_squared_error(y_val, y_pred)
            fold_r2 = r2_score(y_val, y_pred)
            
            metrics['mae'].append(fold_mae)
            metrics['mse'].append(fold_mse)
            metrics['r2'].append(fold_r2)
            
            # Update progress bar description with current metrics
            fold_iterator.set_postfix({
                'MAE': f'{fold_mae:.4f}',
                'MSE': f'{fold_mse:.4f}',
                'R2': f'{fold_r2:.4f}'
            })
            
            print(
                f"Fold {fold}/{cv_folds} - "
                f"MAE: {metrics['mae'][-1]:.4f}, "
                f"MSE: {metrics['mse'][-1]:.4f}, "
                f"R2: {metrics['r2'][-1]:.4f}"
            )
        
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in metrics.items()
        }

    def predict_route_risks(
        self,
        route_segments: List[RouteSegment],
        current_weather: Dict[str, float],
        time_info: Optional[datetime] = None
    ) -> Dict[str, Union[float, List[Dict[str, float]]]]:
        """
        Predict accident risks for a route composed of multiple segments.
        Updated to properly handle categorical features during prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            segment_predictions = []
            high_risk_segments = []
            
            # Use current time if not provided
            if time_info is None:
                time_info = datetime.now(timezone.utc)
            
            for segment in route_segments:
                # Create feature dictionary for segment
                segment_features = {
                    'Start_Lat': segment.start_lat,
                    'Start_Lng': segment.start_lng,
                    'Distance(mi)': segment.distance,
                    'Traffic_Signal': segment.traffic_signals > 0,
                    'Junction': segment.intersections > 0,
                    'Start_Time': time_info,
                    **current_weather  # Unpack weather conditions
                }
                
                # Convert to DataFrame
                segment_df = pd.DataFrame([segment_features])
                
                # Process categorical features using stored mappings
                for feature in self.categorical_features:
                    if feature in segment_df.columns:
                        if feature not in self.label_encoders:
                            print(f"No encoder found for {feature}, using default value")
                            segment_df[feature] = 0  # Use default encoded value
                            continue
                            
                        # Get the feature value
                        value = segment_df[feature].iloc[0]
                        
                        # If value not in mapping, use most common category
                        if value not in self.label_encoders[feature].classes_:
                            print(
                                f"Unknown category '{value}' in feature {feature}, "
                                "using most common category"
                            )
                            value = self.label_encoders[feature].classes_[0]
                        
                        # Encode the value
                        segment_df[feature] = self.label_encoders[feature].transform([value])
                
                # Preprocess segment data
                processed_features = self.preprocess_data(
                    segment_df,
                    train_mode=False
                )
                
                # Ensure all required features are present
                missing_features = set(self.feature_names) - set(processed_features.columns)
                if missing_features:
                    for feature in missing_features:
                        processed_features[feature] = 0  # Use default value
                
                # Reorder columns to match training data
                processed_features = processed_features[self.feature_names]
                
                # Make predictions using both models
                rf_pred = self.rf_model.predict(processed_features)[0]
                gb_pred = self.gb_model.predict(processed_features)[0]
                
                # Combine predictions with weights
                risk_score = 0.6 * rf_pred + 0.4 * gb_pred
                
                # Store prediction results
                segment_result = {
                    'start_lat': segment.start_lat,
                    'start_lng': segment.start_lng,
                    'end_lat': segment.end_lat,
                    'end_lng': segment.end_lng,
                    'distance': segment.distance,
                    'risk_score': risk_score,
                    'risk_factors': self._identify_risk_factors(
                        segment_features,
                        risk_score
                    )
                }
                
                segment_predictions.append(segment_result)
                
                if risk_score > 0.7:
                    high_risk_segments.append(segment_result)
            
            # Calculate overall route risk
            total_distance = sum(segment.distance for segment in route_segments)
            weighted_risk = sum(
                pred['risk_score'] * (pred['distance'] / total_distance)
                for pred in segment_predictions
            )
            
            return {
                'overall_risk': weighted_risk,
                'segment_risks': segment_predictions,
                'high_risk_segments': high_risk_segments,
                'route_length': total_distance,
                'risk_summary': self._generate_risk_summary(segment_predictions)
            }
            
        except Exception as e:
            print(f"Error predicting route risks: {str(e)}")
            raise
    def _identify_risk_factors(
        self,
        features: Dict[str, float],
        risk_score: float
    ) -> Dict[str, float]:
        """
        Identify main factors contributing to risk score.
        
        Parameters:
        features: Dictionary of feature values
        risk_score: Predicted risk score
        
        Returns:
        Dictionary of risk factors and their contributions
        """
        risk_factors = {}
        
        # Analyze weather risks
        if features.get('Weather_Condition') != 'Clear':
            risk_factors['weather'] = self.weather_severity.get(
                features['Weather_Condition'],
                0.5
            )
        
        if features.get('Visibility(mi)', 10) < 5:
            risk_factors['visibility'] = (
                1 - (features['Visibility(mi)'] / 10)
            ) * 0.8
        
        # Check time-based risks
        if features.get('is_rush_hour'):
            risk_factors['rush_hour'] = 0.6
        
        if features.get('Sunrise_Sunset') == 'Night':
            risk_factors['night_driving'] = 0.5
        
        # Infrastructure risks
        if features.get('Junction'):
            risk_factors['junction'] = 0.4
        
        if features.get('Traffic_Signal'):
            risk_factors['traffic_signal'] = 0.3
        
        return dict(sorted(
            risk_factors.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def _generate_risk_summary(
        self,
        segment_predictions: List[Dict[str, float]]
    ) -> Dict[str, Union[float, str]]:
        """
        Generate a summary of route risks with recommendations.
        
        Parameters:
        segment_predictions: List of segment prediction results
        
        Returns:
        Dictionary containing risk summary and recommendations
        """
        high_risk_count = sum(
            1 for segment in segment_predictions
            if segment['risk_score'] > 0.7
        )
        
        avg_risk = np.mean([s['risk_score'] for s in segment_predictions])
        max_risk = max(s['risk_score'] for s in segment_predictions)
        
        risk_factors = []
        for segment in segment_predictions:
            risk_factors.extend(segment['risk_factors'].keys())
        
        common_factors = pd.Series(risk_factors).value_counts().head(3)
        
        return {
            'average_risk': avg_risk,
            'max_risk': max_risk,
            'high_risk_segments_count': high_risk_count,
            'primary_risk_factors': common_factors.index.tolist(),
            'recommendations': self._generate_recommendations(
                avg_risk,
                dict(common_factors)
            )
        }

    def _generate_recommendations(
        self,
        risk_level: float,
        risk_factors: Dict[str, int]
    ) -> List[str]:
        """
        Generate specific safety recommendations based on risk analysis.
        Enhanced to provide more comprehensive recommendations at all risk levels.
        
        Parameters:
        risk_level: Overall risk score
        risk_factors: Dictionary of risk factors and their frequencies
        
        Returns:
        List of safety recommendations
        """
        recommendations = []
        
        # Base recommendations based on risk level
        if risk_level > 0.7:
            recommendations.append(
                "HIGH RISK ALERT: Consider alternative routes or postponing travel if possible."
            )
        elif risk_level > 0.5:
            recommendations.append(
                "MODERATE RISK: Proceed with increased caution and maintain safe distances."
            )
        else:
            recommendations.append(
                "LOW-MODERATE RISK: Standard safety practices recommended."
            )
        
        # Factor-specific recommendations
        for factor, count in risk_factors.items():
            if factor == 'weather':
                recommendations.append(
                    "Monitor weather conditions and adjust driving behavior accordingly."
                )
            elif factor == 'visibility':
                recommendations.append(
                    "Use appropriate vehicle lighting and maintain increased following distance."
                )
            elif factor == 'rush_hour':
                recommendations.append(
                    "Expected traffic during peak hours - plan extra travel time."
                )
            elif factor == 'night_driving':
                recommendations.append(
                    "Ensure proper vehicle lighting and maintain heightened awareness."
                )
        
        # Add general safety recommendations if list is too short
        if len(recommendations) < 2:
            recommendations.extend([
                "Maintain safe following distance from other vehicles.",
                "Stay alert and avoid distractions while driving.",
                f"Current route risk score: {risk_level:.3f} - follow standard safety protocols."
            ])
        
        return recommendations

def load_accident_data(file_path: str) -> pd.DataFrame:
    """
    Load accident data with a progress bar.
    
    Parameters:
    file_path: Path to the CSV file
    
    Returns:
    DataFrame containing the accident data
    """
    # First, count the total lines in the file
    total_lines = sum(1 for _ in open(file_path, 'r'))
    
    # Create a progress bar for data loading
    pbar = tqdm(
        total=total_lines,
        desc="Loading accident data",
        unit="rows"
    )
    
    # Load the data in chunks with progress updates
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=10000):
        chunks.append(chunk)
        pbar.update(len(chunk))
    
    pbar.close()
    return pd.concat(chunks, ignore_index=True)

if __name__ == "__main__":
    print("Loading and processing accident data...")
    df = load_accident_data('/kaggle/input/mini-us-accident/downsized.csv')
    # Add these diagnostic lines here, right after loading the data
    print("\nExploring categorical values in the dataset:")
    print("Unique states in training data:", df['State'].unique())
    print("Unique weather conditions:", df['Weather_Condition'].unique())

    
    # Initialize and train model
    model = AccidentRiskPredictor()
    training_results = model.train(df)
    # Now you can access much more detailed information
    print("\nTraining Results:")
    print(f"Number of samples: {training_results['data_statistics']['n_samples']}")
    print(f"Number of features: {training_results['data_statistics']['n_features']}")

    print("\nModel Performance:")
    print(f"RandomForest MAE: {training_results['final_metrics']['rf_mae']:.4f}")
    print(f"GradientBoosting MAE: {training_results['final_metrics']['gb_mae']:.4f}")

    print("\nTop 5 most important features:")
    for feature, importance in list(training_results['feature_importance'].items())[:5]:
        print(f"{feature}: {importance:.4f}")

    loaded_model = AccidentRiskPredictor.load_model(training_results['model_saved_to'])

    # Example route segments
    route_segments = [
        RouteSegment(
            start_lat=40.7589,  # Manhattan, NY area
            start_lng=-73.9851,
            end_lat=40.7829,
            end_lng=-73.9654,
            distance=3.2,        # Longer segment
            road_type="highway",
            traffic_signals=5,   # High number of traffic signals
            intersections=7      # Many intersections
        ),
        RouteSegment(
            start_lat=40.7829,
            start_lng=-73.9654,
            end_lat=40.8062,
            end_lng=-73.9453,
            distance=2.8,
            road_type="urban",
            traffic_signals=6,
            intersections=8
        )
    ]

    # Create adverse weather conditions
    # Combining multiple risk factors:
    # 1. Poor visibility
    # 2. Precipitation
    # 3. Low temperature (near freezing)
    # 4. High winds

    current_weather = {
        # Weather features
        'Temperature(F)': 33.0,         # Near freezing
        'Wind_Chill(F)': 25.0,         # Very cold wind chill
        'Humidity(%)': 90.0,           # High humidity
        'Pressure(in)': 29.5,          # Low pressure system
        'Visibility(mi)': 2.0,         # Poor visibility
        'Wind_Direction': 'NE',        # Strong northeastern wind
        'Wind_Speed(mph)': 25.0,       # High winds
        'Precipitation(in)': 0.3,      # Active precipitation
        'Weather_Condition': 'Heavy Snow',  # Severe weather

        # Infrastructure features (all default to False unless relevant)
        'Amenity': False,
        'Bump': True,                  # Road hazard
        'Crossing': True,              # Pedestrian crossing
        'Give_Way': True,
        'Junction': True,              # Complex junction
        'No_Exit': False,
        'Railway': True,               # Railway crossing
        'Roundabout': False,
        'Station': True,
        'Stop': True,
        'Traffic_Calming': True,
        'Traffic_Signal': True,
        'Turning_Loop': False,

        # Time-based features - set for night conditions
        'Sunrise_Sunset': 'Night',
        'Civil_Twilight': 'Night',
        'Nautical_Twilight': 'Night',
        'Astronomical_Twilight': 'Night',

        # Location features (will be overwritten by segment data)
        'Start_Lat': 40.7589,
        'Start_Lng': -73.9851,

        # Categorical features
        'State': 'NY',                 # New York
        'Wind_Direction': 'NE',        # Northeastern wind

        # Other features
        'Distance(mi)': 3.2   }         # Longer distance    
    # Predict route risks
    route_risks = loaded_model.predict_route_risks(
        route_segments,
        current_weather
    )
    
    # Print results
    print("\nRoute Risk Analysis:")
    print(f"Overall Risk Score: {route_risks['overall_risk']:.3f}")
    print(f"Route Length: {route_risks['route_length']:.1f} miles")
    print("\nHigh Risk Segments:")
    for segment in route_risks['high_risk_segments']:
        print(
            f"- Segment at ({segment['start_lat']:.4f}, {segment['start_lng']:.4f})"
            f" Risk Score: {segment['risk_score']:.3f}"
        )
    
    print("\nRecommendations:")
    for rec in route_risks['risk_summary']['recommendations']:
        print(f"- {rec}")





