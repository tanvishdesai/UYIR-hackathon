import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple, Optional
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F

# Set up logging with more detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_hazard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    @staticmethod
    def save_config(config: dict, config_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

def generate_synthetic_weather_data(num_samples: int = 1000,
                                  seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic weather data with realistic patterns and correlations
    
    Args:
        num_samples: Number of data points to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame containing synthetic weather data
    """
    np.random.seed(seed)
    
    # Generate timestamps with seasonal patterns
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=x) for x in range(num_samples)]
    
    # Create seasonal temperature variations
    time_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
    base_temp = 65 + 20 * np.sin(2 * np.pi * time_of_year / 365)
    temperature = base_temp + np.random.normal(0, 5, num_samples)
    
    # Generate correlated weather features
    wind_speed = np.abs(np.random.normal(10, 5, num_samples))
    
    # Make precipitation correlate with temperature
    precipitation_prob = 1 / (1 + np.exp((temperature - 70) / 10))
    precipitation = np.random.exponential(0.1, num_samples) * precipitation_prob
    
    # Make visibility correlate inversely with precipitation
    visibility = np.clip(10 - precipitation * 2 + np.random.normal(0, 1, num_samples), 0, 10)
    
    # Create severity levels based on weather conditions
    severity = np.ones(num_samples, dtype=int)
    
    # Define severity rules
    severity[wind_speed > 20] += 1  # High winds
    severity[precipitation > 0.5] += 1  # Heavy precipitation
    severity[visibility < 5] += 1  # Poor visibility
    severity[temperature > 90] += 1  # Extreme heat
    severity[temperature < 32] += 1  # Freezing conditions
    
    # Clip severity to valid range (1-4)
    severity = np.clip(severity, 1, 4)
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature(F)': temperature,
        'Wind_Speed(mph)': wind_speed,
        'Precipitation(in)': precipitation,
        'Visibility(mi)': visibility,
        'Severity': severity
    })

class WeatherHazardDataset(Dataset):
    """Dataset class for weather hazard data with improved handling of class weights"""
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> np.ndarray:
        """Calculate balanced class weights using square root scaling"""
        class_counts = torch.bincount(self.targets)
        class_weights = 1.0 / np.sqrt(class_counts.numpy())
        return class_weights[self.targets]
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
    
    def get_sample_weights(self) -> np.ndarray:
        return self.weights

class WeatherHazardModel(nn.Module):
    """Enhanced neural network for weather hazard prediction with residual connections"""
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_sizes: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3):
        super(WeatherHazardModel, self).__init__()
        
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Create network layers with residual connections
        self.layers = nn.ModuleList()
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            block = nn.Sequential(
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(block)
            
            # Add residual connection if sizes match
            if current_size == hidden_size:
                self.layers.append(lambda x: x)  # Identity residual connection
                
            current_size = hidden_size
        
        self.output = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                x = x + layer(x) if x.size() == layer(x).size() else layer(x)
            else:
                x = layer(x)
        
        logits = self.output(x)
        return F.softmax(logits, dim=1) if not self.training else logits

class WeatherHazardPredictor:
    """Enhanced main class for weather hazard prediction with improved preprocessing and training"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config.load_config(config_path) if config_path else {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = KNNImputer(n_neighbors=5)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized WeatherHazardPredictor using device: {self.device}")
    
    def _handle_missing_data(self, df: pd.DataFrame, 
                            numerical_features: List[str]) -> pd.DataFrame:
        """Handle missing values using KNN imputation with error handling"""
        try:
            logger.info("Handling missing values...")
            df_imputed = df.copy()
            df_imputed[numerical_features] = self.imputer.fit_transform(
                df[numerical_features]
            )
            return df_imputed
        except Exception as e:
            logger.error(f"Error during missing data handling: {e}")
            raise
    
    def _handle_skewed_features(self, df: pd.DataFrame, 
                              columns: List[str]) -> pd.DataFrame:
        """Handle skewed features with robust transformation"""
        try:
            logger.info("Processing skewed features...")
            df_transformed = df.copy()
            
            for col in columns:
                # Handle negative values
                df_transformed[col] = df_transformed[col].clip(lower=0)
                
                # Apply custom transformation based on feature distribution
                if col == 'Precipitation(in)':
                    df_transformed[col] = df_transformed[col].replace(0, 0.01)
                    df_transformed[col] = np.log1p(df_transformed[col])
                else:
                    # Use Box-Cox transformation for other features
                    positive_values = df_transformed[col] + 1e-6
                    df_transformed[col] = np.log1p(positive_values)
            
            return df_transformed
        except Exception as e:
            logger.error(f"Error during skewed feature handling: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
        """Preprocess weather data with enhanced feature engineering"""
        try:
            # Define feature groups
            numerical_features = ['Temperature(F)', 'Wind_Speed(mph)', 
                                'Precipitation(in)', 'Visibility(mi)']
            skewed_features = ['Precipitation(in)', 'Visibility(mi)']
            
            # Handle missing and skewed data
            df = self._handle_missing_data(df, numerical_features)
            df = self._handle_skewed_features(df, skewed_features)
            
            # Add time-based features
            df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
            df['DayOfWeek'] = pd.to_datetime(df['Timestamp']).dt.weekday
            df['Month'] = pd.to_datetime(df['Timestamp']).dt.month
            df['Season'] = pd.to_datetime(df['Timestamp']).dt.month % 12 // 3
            
            # Create interaction features
            df['Temp_Wind_Interaction'] = df['Temperature(F)'] * df['Wind_Speed(mph)']
            df['Precip_Vis_Interaction'] = df['Precipitation(in)'] * df['Visibility(mi)']
            
            # One-hot encode categorical variables
            df = pd.get_dummies(df, columns=['Hour', 'DayOfWeek', 'Month', 'Season'])
            
            # Scale numerical features
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            
            # Prepare features and target
            feature_columns = (numerical_features + 
                             ['Temp_Wind_Interaction', 'Precip_Vis_Interaction'] +
                             [col for col in df.columns if col.startswith(
                                 ('Hour_', 'DayOfWeek_', 'Month_', 'Season_'))])
            
            X = df[feature_columns].values
            y = self.label_encoder.fit_transform(df['Severity'])
            
            # Apply SMOTE for handling class imbalance
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            return train_test_split(X_resampled, y_resampled, 
                                  test_size=0.2, stratify=y_resampled, 
                                  random_state=42)
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise

    def train(self, df: pd.DataFrame, batch_size: int = 128, 
              epochs: int = 50, learning_rate: float = 0.001, 
              patience: int = 5) -> None:
        """Train the weather hazard prediction model with enhanced monitoring"""
        try:
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            
            # Create datasets and data loaders
            train_dataset = WeatherHazardDataset(X_train, y_train)
            test_dataset = WeatherHazardDataset(X_test, y_test)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=WeightedRandomSampler(
                    weights=train_dataset.get_sample_weights(),
                    num_samples=len(train_dataset),
                    replacement=True
                )
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Initialize model
            num_classes = len(np.unique(y_train))
            input_size = X_train.shape[1]
            self.model = WeatherHazardModel(input_size, num_classes).to(self.device)
            
            # Training setup with gradient clipping
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(self.model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=1e-5)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                                        factor=0.1, patience=patience//2)
            
            self._train_loop(train_loader, test_loader, criterion, 
                           optimizer, scheduler, epochs, patience)
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def _train_loop(self, train_loader: DataLoader, 
                   test_loader: DataLoader, 
                   criterion: nn.Module, 
                   optimizer: optim.Optimizer, 
                   scheduler: ReduceLROnPlateau, 
                   epochs: int, 
                   patience: int) -> None:
        """Training loop with enhanced monitoring and visualization"""
        metrics = defaultdict(list)
        best_f1 = 0
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, all_predictions, all_labels = self._validate_epoch(
                test_loader, criterion)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            
 # Update metrics
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            
            # Learning rate scheduling
            scheduler.step(f1)
            
            # Save training state for potential recovery
            self._save_checkpoint(epoch, metrics, optimizer)
            
            # Early stopping check with model saving
            if f1 > best_f1:
                best_f1 = f1
                epochs_without_improvement = 0
                self._save_model(metrics, 'best_weather_hazard_model.pth')
                logger.info(f"New best model saved with F1 score: {f1:.4f}")
            else:
                epochs_without_improvement += 1
            
            self._log_epoch_metrics(epoch, epochs, train_loss, val_loss, f1)
            
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered - no improvement in F1 score")
                break
        
        # Generate final visualizations and reports
        self._plot_metrics(metrics)
        self._plot_confusion_matrix(all_labels, all_predictions)
        self._generate_training_report(metrics)
        
        # Load the best model for future predictions
        self._load_best_model()

    def _train_epoch(self, train_loader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: optim.Optimizer) -> float:
        """Execute one epoch of training with gradient clipping"""
        self.model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            try:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
            except RuntimeError as e:
                logger.error(f"Runtime error during training: {e}")
                raise
                
        return total_loss / len(train_loader)

    def _validate_epoch(self, test_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, List, List]:
        """Execute one epoch of validation with uncertainty estimation"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        prediction_uncertainties = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                try:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    total_loss += loss.item()
                    
                    # Calculate prediction uncertainty using softmax probabilities
                    probabilities = F.softmax(outputs, dim=1)
                    uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                    prediction_uncertainties.extend(uncertainty.cpu().numpy())
                    
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    
                except RuntimeError as e:
                    logger.error(f"Runtime error during validation: {e}")
                    raise
        
        # Store uncertainty metrics for analysis
        self._store_uncertainty_metrics(prediction_uncertainties, all_predictions, all_labels)
        
        return total_loss / len(test_loader), all_predictions, all_labels

    def _save_checkpoint(self, epoch: int, metrics: Dict, 
                        optimizer: optim.Optimizer) -> None:
        """Save training checkpoint for potential recovery"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.model_dir / 'checkpoint.pth')

    def _save_model(self, metrics: Dict, filename: str) -> None:
        """Save model with metadata and configurations"""
        model_info = {
            'model_state': self.model.state_dict(),
            'scaler_state': self.scaler.__dict__,
            'label_encoder_state': self.label_encoder.__dict__,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        torch.save(model_info, self.model_dir / filename)

    def _load_best_model(self) -> None:
        """Load the best model with error handling"""
        try:
            model_path = self.model_dir / 'best_weather_hazard_model.pth'
            model_info = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_info['model_state'])
            logger.info("Successfully loaded best model")
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            raise

    def _generate_training_report(self, metrics: Dict) -> None:
        """Generate comprehensive training report with metrics and visualizations"""
        report = {
            'training_summary': {
                'final_f1': metrics['f1'][-1],
                'best_f1': max(metrics['f1']),
                'final_precision': metrics['precision'][-1],
                'final_recall': metrics['recall'][-1],
                'epochs_trained': len(metrics['f1']),
                'training_completed': datetime.now().isoformat()
            },
            'model_configuration': self.config,
            'performance_metrics': {
                'f1_scores': metrics['f1'],
                'precision_scores': metrics['precision'],
                'recall_scores': metrics['recall']
            }
        }
        
        # Save report as JSON
        with open(self.model_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=4)

    def predict(self, features: np.ndarray, return_uncertainty: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with optional uncertainty estimation"""
        try:
            self.model.eval()
            with torch.no_grad():
                features = torch.FloatTensor(features).to(self.device)
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
                
                # Calculate prediction uncertainty
                uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                predictions = self.label_encoder.inverse_transform(predicted.cpu().numpy())
                
                if return_uncertainty:
                    return predictions, uncertainty.cpu().numpy()
                return predictions
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

def preprocess_accident_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess US Accident dataset with enhanced feature engineering and robust datetime handling
    
    This function now includes improved datetime parsing to handle timestamps with milliseconds
    and various formats that might be present in the dataset.
    
    Args:
        df: Raw US Accident dataset DataFrame
    
    Returns:
        Preprocessed DataFrame ready for accident severity prediction
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Convert timestamp columns to datetime with robust parsing
    try:
        # First attempt: Try parsing with ISO format
        df_processed['Start_Time'] = pd.to_datetime(
            df_processed['Start_Time'],
            format='ISO8601',
            errors='coerce'
        )
        df_processed['End_Time'] = pd.to_datetime(
            df_processed['End_Time'],
            format='ISO8601',
            errors='coerce'
        )
        
        # Check for any failed conversions (NaT values)
        if df_processed['Start_Time'].isna().any() or df_processed['End_Time'].isna().any():
            logger.warning("Some datetime conversions failed. Attempting alternative parsing...")
            
            # Second attempt: Try parsing with mixed format
            mask_start = df_processed['Start_Time'].isna()
            mask_end = df_processed['End_Time'].isna()
            
            if mask_start.any():
                df_processed.loc[mask_start, 'Start_Time'] = pd.to_datetime(
                    df.loc[mask_start, 'Start_Time'],
                    format='mixed',
                    errors='coerce'
                )
            
            if mask_end.any():
                df_processed.loc[mask_end, 'End_Time'] = pd.to_datetime(
                    df.loc[mask_end, 'End_Time'],
                    format='mixed',
                    errors='coerce'
                )
        
        # Remove rows where datetime conversion failed
        invalid_dates = df_processed['Start_Time'].isna() | df_processed['End_Time'].isna()
        if invalid_dates.any():
            logger.warning(f"Removing {invalid_dates.sum()} rows with invalid dates")
            df_processed = df_processed[~invalid_dates]
        
    except Exception as e:
        logger.error(f"Error during datetime conversion: {e}")
        raise
    
    # Calculate accident duration in minutes with validation
    df_processed['Duration_Minutes'] = (
        (df_processed['End_Time'] - df_processed['Start_Time'])
        .dt.total_seconds() / 60
    )
    
    # Extract temporal features with error checking
    try:
        df_processed['Hour'] = df_processed['Start_Time'].dt.hour
        df_processed['Day_Of_Week'] = df_processed['Start_Time'].dt.dayofweek
        df_processed['Month'] = df_processed['Start_Time'].dt.month
        df_processed['Is_Weekend'] = df_processed['Day_Of_Week'].isin([5, 6]).astype(int)
        df_processed['Year'] = df_processed['Start_Time'].dt.year
        
        # Create time-of-day categories
        df_processed['Time_Of_Day'] = pd.cut(
            df_processed['Hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
    except Exception as e:
        logger.error(f"Error during temporal feature extraction: {e}")
        raise
    
    # Calculate distance between start and end points
    df_processed['Location_Change'] = np.where(
        (df_processed['Start_Lat'] != df_processed['End_Lat']) |
        (df_processed['Start_Lng'] != df_processed['End_Lng']),
        1, 0
    )
    
    # Calculate center point of accident
    df_processed['Center_Lat'] = (
        df_processed['Start_Lat'] + df_processed['End_Lat']
    ) / 2
    df_processed['Center_Lng'] = (
        df_processed['Start_Lng'] + df_processed['End_Lng']
    ) / 2
    
    # Create feature indicating if accident affects a long distance
    df_processed['Is_Long_Distance'] = (
        df_processed['Distance(mi)'] > df_processed['Distance(mi)'].median()
    ).astype(int)
    
    # Handle missing values
    df_processed['Duration_Minutes'].fillna(
        df_processed['Duration_Minutes'].median(),
        inplace=True
    )
    df_processed['Distance(mi)'].fillna(
        df_processed['Distance(mi)'].median(),
        inplace=True
    )
    
    # Remove obvious outliers
    df_processed = df_processed[
        (df_processed['Duration_Minutes'] >= 0) &
        (df_processed['Duration_Minutes'] <= 24 * 60) &  # Max 24 hours
        (df_processed['Distance(mi)'] >= 0) &
        (df_processed['Distance(mi)'] <= df_processed['Distance(mi)'].quantile(0.99))  # Remove extreme distances
    ]
    
    # Create final feature set
    feature_columns = [
        'Duration_Minutes',
        'Distance(mi)',
        'Hour',
        'Day_Of_Week',
        'Month',
        'Year',  # Added year as a feature
        'Is_Weekend',
        'Location_Change',
        'Is_Long_Distance',
        'Center_Lat',
        'Center_Lng'
    ]
    
    # Add one-hot encoded categorical variables
    df_processed = pd.get_dummies(
        df_processed,
        columns=['Time_Of_Day'],
        prefix=['TimeOfDay']
    )
    
    # Combine all features
    feature_columns.extend([
        col for col in df_processed.columns
        if col.startswith('TimeOfDay_')
    ])
    
    # Log summary statistics
    logger.info(f"Processed {len(df_processed)} accidents")
    logger.info(f"Feature set includes {len(feature_columns)} features")
    
    # Prepare final dataset
    return df_processed[feature_columns + ['Severity']]
    
def main():
    """Main function for enhanced accident severity prediction"""
    try:
        # Load the US Accident dataset
        logger.info("Loading US Accident dataset...")
        df = pd.read_csv(r'us-accidents\US_Accidents_March23.csv')
        
        # Preprocess the accident dataset
        logger.info("Preprocessing accident data...")
        df_processed = preprocess_accident_data(df)
        
        # Initialize predictor with enhanced configuration
        config = {
            'model_params': {
                'hidden_sizes': [256, 128, 64],
                'dropout_rate': 0.4
            },
            'training_params': {
                'batch_size': 512,  # Larger batch size for stability
                'learning_rate': 0.001,
                'patience': 10,     # Increased patience for complex patterns
                'epochs': 50        # More epochs for better convergence
            }
        }
        
        logger.info("Initializing predictor with configuration...")
        predictor = WeatherHazardPredictor()
        predictor.config = config
        
        logger.info("Starting model training...")
        predictor.train(
            df_processed,
            batch_size=config['training_params']['batch_size'],
            learning_rate=config['training_params']['learning_rate'],
            patience=config['training_params']['patience'],
            epochs=config['training_params']['epochs']
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()