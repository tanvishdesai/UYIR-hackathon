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
from tqdm import tqdm
import time

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

class ProgressTracker:
    """Utility class for tracking and displaying progress"""
    def __init__(self, total_steps: int, description: str = ""):
        self.progress_bar = tqdm(total=total_steps, desc=description)
        self.start_time = time.time()
        
    def update(self, steps: int = 1):
        self.progress_bar.update(steps)
        
    def set_description(self, desc: str):
        self.progress_bar.set_description(desc)
        
    def close(self):
        self.progress_bar.close()
        
    def get_elapsed_time(self) -> str:
        elapsed = time.time() - self.start_time
        return time.strftime("%H:%M:%S", time.gmtime(elapsed))

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
        
        self.input_norm = nn.BatchNorm1d(input_size)
        
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
        return torch.softmax(logits, dim=1) if not self.training else logits

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
        """Preprocess the accident data with progress tracking"""
        logger.info("Starting data preprocessing...")
        progress = ProgressTracker(total_steps=7, description="Preprocessing")
        
        df_processed = df.copy()
        
        # Process timestamps
        progress.set_description("Processing timestamps")
        timestamp_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
        for col in timestamp_columns:
            df_processed[col] = pd.to_datetime(df_processed[col], format='mixed', errors='coerce')
            df_processed[col] = df_processed[col].ffill()  # Using ffill() as recommended
        progress.update()
        
        # Extract temporal features
        progress.set_description("Extracting temporal features")
        df_processed['Duration_Minutes'] = (
            (df_processed['End_Time'] - df_processed['Start_Time'])
            .dt.total_seconds() / 60
        )
        
        duration_mask = (
            (df_processed['Duration_Minutes'] < 0) | 
            (df_processed['Duration_Minutes'] > 24*60)
        )
        if duration_mask.any():
            median_duration = df_processed['Duration_Minutes'].median()
            df_processed.loc[duration_mask, 'Duration_Minutes'] = median_duration
            
        df_processed['Hour'] = df_processed['Start_Time'].dt.hour
        df_processed['Day_Of_Week'] = df_processed['Start_Time'].dt.dayofweek
        df_processed['Month'] = df_processed['Start_Time'].dt.month
        df_processed['Is_Weekend'] = df_processed['Day_Of_Week'].isin([5, 6]).astype(int)
        df_processed['Is_Rush_Hour'] = df_processed['Hour'].isin([7,8,9,16,17,18]).astype(int)
        progress.update()
        
        # Process weather features
        progress.set_description("Processing weather features")
        weather_features = [
            'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 
            'Precipitation(in)'
        ]
        df_processed[weather_features] = self.imputer.fit_transform(
            df_processed[weather_features]
        )
        progress.update()
        
        # Create weather indicators
        progress.set_description("Creating weather indicators")
        df_processed['Poor_Visibility'] = (df_processed['Visibility(mi)'] < 2).astype(int)
        df_processed['Heavy_Precipitation'] = (df_processed['Precipitation(in)'] > 0.3).astype(int)
        df_processed['Extreme_Temperature'] = (
            (df_processed['Temperature(F)'] > 90) | 
            (df_processed['Temperature(F)'] < 32)
        ).astype(int)
        progress.update()
        
        # Process road features
        progress.set_description("Processing road features")
        road_features = [
            'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
            'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
            'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
        ]
        for feature in road_features:
            df_processed[feature] = df_processed[feature].astype(int)
        progress.update()
        
        # Calculate distance features
        progress.set_description("Calculating distance features")
        df_processed['Distance_From_Start_To_End'] = np.sqrt(
            (df_processed['End_Lat'] - df_processed['Start_Lat'])**2 +
            (df_processed['End_Lng'] - df_processed['Start_Lng'])**2
        )
        
        light_conditions = [
            'Sunrise_Sunset', 'Civil_Twilight', 
            'Nautical_Twilight', 'Astronomical_Twilight'
        ]
        for condition in light_conditions:
            df_processed[f'Is_{condition}_Night'] = (
                df_processed[condition] == 'Night'
            ).astype(int)
        progress.update()
        
        # Final feature selection and scaling
        progress.set_description("Scaling features")
        feature_columns = (
            weather_features +
            road_features +
            ['Duration_Minutes', 'Distance(mi)', 'Distance_From_Start_To_End',
             'Hour', 'Day_Of_Week', 'Month', 'Is_Weekend', 'Is_Rush_Hour',
             'Poor_Visibility', 'Heavy_Precipitation', 'Extreme_Temperature'] +
            [col for col in df_processed.columns if col.startswith('Is_') and 
             col not in ['Is_Weekend', 'Is_Rush_Hour']]
        )
        
        numerical_features = [
            col for col in feature_columns 
            if df_processed[col].dtype in ['float64', 'int64']
        ]
        df_processed[numerical_features] = self.scaler.fit_transform(
            df_processed[numerical_features]
        )
        progress.update()
        
        progress.close()
        logger.info(f"Preprocessing completed in {progress.get_elapsed_time()}")
        
        return df_processed[feature_columns + ['Severity']]

    def train(self, df: pd.DataFrame, batch_size: int = 256, 
              epochs: int = 50, learning_rate: float = 0.001,
              patience: int = 10) -> None:
        """Train the model with progress tracking"""
        try:
            # Preprocess data
            processed_data = self.preprocess_data(df)
            X = processed_data.drop('Severity', axis=1).values
            y = processed_data['Severity'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Create datasets and loaders
            train_dataset = AccidentDataset(X_train, y_train)
            test_dataset = AccidentDataset(X_test, y_test)
            
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
        """Training loop with progress tracking"""
        best_f1 = 0
        epochs_without_improvement = 0
        history = {'train_loss': [], 'val_loss': [], 'f1': []}
        
        epoch_progress = ProgressTracker(total_steps=epochs, description="Training")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            batch_progress = ProgressTracker(
                total_steps=len(train_loader),
                description=f"Epoch {epoch+1}/{epochs}"
            )
            
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
                batch_progress.update()
            
            batch_progress.close()
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, f1_score = self._validate(test_loader, criterion)
            
            # Update learning rate
            scheduler.step(f1_score)
            
            # Save metrics
# Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['f1'].append(f1_score)
            
            # Update progress description with metrics
            epoch_progress.set_description(
                f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - F1: {f1_score:.4f}"
            )
            epoch_progress.update()
            
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
        
        epoch_progress.close()
        logger.info(f"Training completed in {epoch_progress.get_elapsed_time()}")
        
        # Load best model
        self._load_best_model()
        
        # Plot training history
        self._plot_training_history(history)

    def _validate(self, test_loader: DataLoader, 
                 criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model with progress tracking"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        val_progress = ProgressTracker(
            total_steps=len(test_loader),
            description="Validation"
        )
        
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
                
                val_progress.update()
        
        val_progress.close()
        
        val_loss = total_loss / len(test_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, 
            all_predictions, 
            average='weighted'
        )
        
        return val_loss, f1

    def _save_model(self, history: Dict, filename: str) -> None:
        """Save model state and training history with metadata"""
        try:
            model_info = {
                'model_state': self.model.state_dict(),
                'scaler_state': self.scaler.__dict__,
                'imputer_state': self.imputer.__dict__,
                'history': history,
                'timestamp': datetime.now().isoformat(),
                'model_config': {
                    'input_size': self.model.input_norm.num_features,
                    'hidden_sizes': [layer.out_features for layer in self.model.hidden_layers if isinstance(layer, nn.Linear)],
                    'device': str(self.device)
                }
            }
            torch.save(model_info, self.model_dir / filename)
            logger.info(f"Model saved successfully: {filename}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def _load_best_model(self) -> None:
        """Load the best performing model with error handling"""
        try:
            model_path = self.model_dir / 'best_accident_model.pth'
            model_info = torch.load(model_path, map_location=self.device)
            
            # Verify model configuration
            saved_config = model_info['model_config']
            if (saved_config['input_size'] != self.model.input_norm.num_features or
                saved_config['hidden_sizes'] != [layer.out_features for layer in self.model.hidden_layers if isinstance(layer, nn.Linear)]):
                raise ValueError("Saved model architecture does not match current model")
            
            self.model.load_state_dict(model_info['model_state'])
            self.scaler.__dict__.update(model_info['scaler_state'])
            self.imputer.__dict__.update(model_info['imputer_state'])
            logger.info("Best model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            raise

    def _plot_training_history(self, history: Dict) -> None:
        """Plot training metrics history with enhanced visualization"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 5))
        
        # Plot losses with shared x-axis
        ax1 = fig.add_subplot(131)
        ax1.plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
        ax1.plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot F1 score
        ax2 = fig.add_subplot(132)
        ax2.plot(history['f1'], label='F1 Score', color='green', alpha=0.7)
        ax2.set_title('F1 Score Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot learning curves
        ax3 = fig.add_subplot(133)
        epochs = range(1, len(history['train_loss']) + 1)
        ax3.plot(epochs, history['train_loss'], 'bo-', label='Training Loss', alpha=0.5)
        ax3.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', alpha=0.5)
        ax3.set_title('Learning Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix with enhanced visualization"""
        plt.style.use('seaborn')
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot both raw counts and percentages
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', alpha=0.7)
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Reds', alpha=0.3)
        
        plt.title('Confusion Matrix\n(Numbers: Raw Counts, Percentages: Normalized)')
        plt.xlabel('Predicted Severity')
        plt.ylabel('True Severity')
        
        plt.savefig(self.model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions on new data with progress tracking"""
        try:
            logger.info("Starting prediction process...")
            
            # Preprocess input data
            processed_data = self.preprocess_data(df)
            features = processed_data.drop('Severity', axis=1).values
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Make predictions in batches to handle large datasets
            batch_size = 1024
            predictions = []
            probabilities = []
            
            self.model.eval()
            with torch.no_grad():
                predict_progress = ProgressTracker(
                    total_steps=(len(features) + batch_size - 1) // batch_size,
                    description="Generating predictions"
                )
                
                for i in range(0, len(features), batch_size):
                    batch = features_tensor[i:i+batch_size]
                    outputs = self.model(batch)
                    probs = torch.softmax(outputs, dim=1)
                    probabilities.append(probs)
                    
                    if not return_proba:
                        _, preds = torch.max(outputs.data, 1)
                        predictions.append(preds)
                        
                    predict_progress.update()
                
                predict_progress.close()
            
            if return_proba:
                return torch.cat(probabilities, dim=0).cpu().numpy()
            
            predictions = torch.cat(predictions, dim=0)
            return predictions.cpu().numpy() + 1  # Convert back to 1-based indexing
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate model performance with detailed metrics"""
        try:
            logger.info("Starting model evaluation...")
            
            # Get predictions
            y_pred = self.predict(test_df)
            y_true = test_df['Severity'].values
            
            # Calculate detailed metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                average=None
            )
            
            # Calculate per-class metrics
            class_metrics = {}
            for i in range(len(precision)):
                class_metrics[f'class_{i+1}'] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': support[i]
                }
            
            # Calculate overall metrics
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='weighted'
            )
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_true, y_pred)
            
            # Compile all metrics
            metrics = {
                'overall': {
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1
                },
                'per_class': class_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save metrics to file
            metrics_path = self.model_dir / 'evaluation_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Evaluation completed. Metrics saved to {metrics_path}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

def main():
    """Main function demonstrating usage with proper error handling"""
    try:
        # Configure progress tracking for main execution
        main_progress = ProgressTracker(total_steps=4, description="Overall Progress")
        
        # Load data
        logger.info("Loading US Accident dataset...")
        df = pd.read_csv(r'us-accidents\US_Accidents_March23.csv')
        main_progress.update()
        
        # Initialize predictor
        predictor = AccidentSeverityPredictor()
        main_progress.update()
        
        # Train model
        logger.info("Starting model training...")
        predictor.train(
            df=df,
            batch_size=256,
            epochs=50,
            learning_rate=0.001,
            patience=10
        )
        main_progress.update()
        
        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics = predictor.evaluate(df)
        
        # Save evaluation results
        results_path = Path('evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        main_progress.update()
        main_progress.close()
        
        logger.info(f"Model training and evaluation completed successfully")
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()