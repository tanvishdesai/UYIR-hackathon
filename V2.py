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
import seaborn as sns
from datetime import datetime
from pathlib import Path
import yaml
import json
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('accident_severity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for model parameters and training settings"""
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

class AccidentDataset(Dataset):
    """Custom Dataset class for handling accident data"""
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets) - 1  # Convert to 0-based indexing
        self.weights = self._calculate_sample_weights()
    
    def _calculate_sample_weights(self) -> np.ndarray:
        """Calculate balanced weights for handling class imbalance"""
        class_counts = torch.bincount(self.targets)
        class_weights = 1.0 / np.sqrt(class_counts.numpy())
        return class_weights[self.targets]
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
    
    def get_sample_weights(self) -> np.ndarray:
        return self.weights

class AccidentSeverityModel(nn.Module):
    """Neural network model for accident severity prediction"""
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_sizes: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.4):
        super(AccidentSeverityModel, self).__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Create network layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        logits = self.output(x)
        return F.softmax(logits, dim=1) if not self.training else logits

class AccidentSeverityPredictor:
    """Main class for accident severity prediction"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config.load_config(config_path) if config_path else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
        logger.info(f"Initialized AccidentSeverityPredictor using device: {self.device}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the accident data with robust datetime handling"""
        logger.info("Starting data preprocessing...")
        df_processed = df.copy()
        
        # Convert timestamps with proper format handling
        timestamp_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
        for col in timestamp_columns:
            try:
                # Use 'coerce' to handle any invalid timestamps by setting them to NaT
                # format='mixed' allows pandas to infer the format for each timestamp
                df_processed[col] = pd.to_datetime(
                    df_processed[col],
                    format='mixed',
                    errors='coerce'
                )
                
                # Check for and handle NaT values
                nat_count = df_processed[col].isna().sum()
                if nat_count > 0:
                    logger.warning(
                        f"Found {nat_count} invalid timestamps in {col}. "
                        "These will be forward-filled where possible."
                    )
                    df_processed[col] = df_processed[col].fillna(method='ffill')
            
            except Exception as e:
                logger.error(f"Error processing timestamp column {col}: {e}")
                raise
        
        # Extract temporal features with error handling
        try:
            df_processed['Duration_Minutes'] = (
                (df_processed['End_Time'] - df_processed['Start_Time'])
                .dt.total_seconds() / 60
            )
            
            # Handle negative or extreme durations
            duration_mask = (
                (df_processed['Duration_Minutes'] < 0) | 
                (df_processed['Duration_Minutes'] > 24*60)  # More than 24 hours
            )
            if duration_mask.any():
                logger.warning(
                    f"Found {duration_mask.sum()} invalid durations. "
                    "Setting to median duration."
                )
                median_duration = df_processed['Duration_Minutes'].median()
                df_processed.loc[duration_mask, 'Duration_Minutes'] = median_duration
            
            # Extract other temporal features
            df_processed['Hour'] = df_processed['Start_Time'].dt.hour
            df_processed['Day_Of_Week'] = df_processed['Start_Time'].dt.dayofweek
            df_processed['Month'] = df_processed['Start_Time'].dt.month
            df_processed['Is_Weekend'] = df_processed['Day_Of_Week'].isin([5, 6]).astype(int)
            df_processed['Is_Rush_Hour'] = df_processed['Hour'].isin([7,8,9,16,17,18]).astype(int)
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            raise
        
        # Process weather features
        weather_features = [
            'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 
            'Precipitation(in)'
        ]
        
        # Handle missing weather data
        df_processed[weather_features] = self.imputer.fit_transform(
            df_processed[weather_features]
        )
        
        # Create weather condition indicators
        df_processed['Poor_Visibility'] = (df_processed['Visibility(mi)'] < 2).astype(int)
        df_processed['Heavy_Precipitation'] = (df_processed['Precipitation(in)'] > 0.3).astype(int)
        df_processed['Extreme_Temperature'] = (
            (df_processed['Temperature(F)'] > 90) | 
            (df_processed['Temperature(F)'] < 32)
        ).astype(int)
        
        # Process road features
        road_features = [
            'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
            'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
            'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
        ]
        
        for feature in road_features:
            df_processed[feature] = df_processed[feature].astype(int)
        
        # Calculate distance features
        df_processed['Distance_From_Start_To_End'] = np.sqrt(
            (df_processed['End_Lat'] - df_processed['Start_Lat'])**2 +
            (df_processed['End_Lng'] - df_processed['Start_Lng'])**2
        )
        
        # Process time period features
        light_conditions = [
            'Sunrise_Sunset', 'Civil_Twilight', 
            'Nautical_Twilight', 'Astronomical_Twilight'
        ]
        for condition in light_conditions:
            df_processed[f'Is_{condition}_Night'] = (
                df_processed[condition] == 'Night'
            ).astype(int)
        
        # Encode weather conditions
        df_processed['Is_Adverse_Weather'] = df_processed['Weather_Condition'].str.contains(
            'Rain|Snow|Fog|Mist|Hail|Thunder|Storm|Windy',
            case=False, na=False
        ).astype(int)
        
        # Select and scale features
        feature_columns = (
            weather_features +
            road_features +
            ['Duration_Minutes', 'Distance(mi)', 'Distance_From_Start_To_End',
             'Hour', 'Day_Of_Week', 'Month', 'Is_Weekend', 'Is_Rush_Hour',
             'Poor_Visibility', 'Heavy_Precipitation', 'Extreme_Temperature',
             'Is_Adverse_Weather'] +
            [col for col in df_processed.columns if col.startswith('Is_') and 
             col not in ['Is_Weekend', 'Is_Rush_Hour', 'Is_Adverse_Weather']]
        )
        
        # Scale numerical features
        numerical_features = [
            col for col in feature_columns 
            if df_processed[col].dtype in ['float64', 'int64']
        ]
        df_processed[numerical_features] = self.scaler.fit_transform(
            df_processed[numerical_features]
        )
        
        return df_processed[feature_columns + ['Severity']]

    def train(self, df: pd.DataFrame, batch_size: int = 256, 
              epochs: int = 50, learning_rate: float = 0.001,
              patience: int = 10) -> None:
        """Train the model"""
        try:
            # Preprocess data
            processed_data = self.preprocess_data(df)
            X = processed_data.drop('Severity', axis=1).values
            y = processed_data['Severity'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Create datasets
            train_dataset = AccidentDataset(X_train, y_train)
            test_dataset = AccidentDataset(X_test, y_test)
            
            # Create data loaders
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
            input_size = X_train.shape[1]
            self.model = AccidentSeverityModel(
                input_size=input_size,
                num_classes=len(np.unique(y)),
                hidden_sizes=[256, 128, 64]
            ).to(self.device)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=patience//2
            )
            
            self._train_loop(
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                patience
            )
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _train_loop(self, train_loader: DataLoader, 
                    test_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: ReduceLROnPlateau,
                    epochs: int,
                    patience: int) -> None:
        """Training loop with monitoring and early stopping"""
        best_f1 = 0
        epochs_without_improvement = 0
        history = {'train_loss': [], 'val_loss': [], 'f1': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, f1_score = self._validate(test_loader, criterion)
            
            # Update learning rate
            scheduler.step(f1_score)
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['f1'].append(f1_score)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"F1 Score: {f1_score:.4f}"
            )
            
            # Check for improvement
            if f1_score > best_f1:
                best_f1 = f1_score
                epochs_without_improvement = 0
                self._save_model(history, 'best_accident_model.pth')
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        self._load_best_model()
        
        # Plot training history
        self._plot_training_history(history)

    def _validate(self, test_loader: DataLoader, 
                 criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model and compute metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_loss = total_loss / len(test_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, 
            all_predictions, 
            average='weighted'
        )
        
        return val_loss, f1

    def _save_model(self, history: Dict, filename: str) -> None:
        """Save model state and training history"""
        model_info = {
            'model_state': self.model.state_dict(),
            'scaler_state': self.scaler.__dict__,
            'imputer_state': self.imputer.__dict__,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(model_info, self.model_dir / filename)
        logger.info(f"Model saved: {filename}")

    def _load_best_model(self) -> None:
        """Load the best performing model"""
        try:
            model_path = self.model_dir / 'best_accident_model.pth'
            model_info = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_info['model_state'])
            self.scaler.__dict__.update(model_info['scaler_state'])
            self.imputer.__dict__.update(model_info['imputer_state'])
            logger.info("Best model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            raise

    def _plot_training_history(self, history: Dict) -> None:
        """Plot training metrics history"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot F1 score
        plt.subplot(1, 2, 2)
        plt.plot(history['f1'], label='F1 Score')
        plt.title('F1 Score Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix of model predictions"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.model_dir / 'confusion_matrix.png')
        plt.close()

    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions on new data"""
        try:
            # Preprocess input data
            processed_data = self.preprocess_data(df)
            features = processed_data.drop('Severity', axis=1).values
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                if return_proba:
                    return probabilities.cpu().numpy()
                
                _, predictions = torch.max(outputs.data, 1)
                # Convert back to original severity levels (1-based)
                return predictions.cpu().numpy() + 1
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data"""
        try:
            # Get predictions
            y_pred = self.predict(test_df)
            y_true = test_df['Severity'].values
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='weighted'
            )
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_true, y_pred)
            
            # Return metrics
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

def main():
    """Main function to demonstrate usage"""
    try:
        # Load data
        logger.info("Loading US Accident dataset...")
        df = pd.read_csv(r'us-accidents\US_Accidents_March23.csv')
        
        # Initialize predictor
        predictor = AccidentSeverityPredictor()
        
        # Train model
        logger.info("Starting model training...")
        predictor.train(
            df=df,
            batch_size=256,
            epochs=50,
            learning_rate=0.001,
            patience=10
        )
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics = predictor.evaluate(df)
        
        # Save evaluation results
        with open('evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()