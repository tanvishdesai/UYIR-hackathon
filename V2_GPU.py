import gc
import os
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
import json
import logging
import faiss
from joblib import Parallel, delayed
from typing import List, Tuple
import cupy as cp  # For GPU acceleration of numerical operations

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

# Default model configuration
MODEL_CONFIG = {
    'model_params': {
        'hidden_sizes': [256, 128, 64],
        'dropout_rate': 0.4,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 128,
        'epochs': 50,
        'patience': 10,
        'grad_clip': 1.0
    },
    'training_params': {
        'test_size': 0.2,
        'random_state': 42
    },
    'gpu_params': {
        'cuda_devices': [0],  # List of GPU devices to use
        'pin_memory': True,
        'num_workers': 2,
        'empty_cache_freq': 10  # New parameter for cache clearing frequency

    }
}

def clear_gpu_cache():
    """Clear GPU cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class MemoryOptimizedDataLoader(DataLoader):
    """Custom DataLoader with memory optimization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_cache_freq = kwargs.get('empty_cache_freq', 10)
        self.batch_count = 0
    
    def __iter__(self):
        for batch in super().__iter__():
            self.batch_count += 1
            if self.batch_count % self.empty_cache_freq == 0:
                clear_gpu_cache()
            yield batch


class GPUKNNImputer:
    def __init__(self, n_neighbors=5, batch_size=1000):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.fitted_values = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        print("Starting GPU KNN imputation...")
        
        # Convert to torch tensor and move to GPU
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        mask = torch.isnan(X)
        
        if not mask.any():
            return X.cpu().numpy()
            
        # Get non-missing rows for fitting
        missing_rows = torch.where(mask.any(dim=1))[0]
        valid_rows = torch.where(~mask.any(dim=1))[0]
        
        if len(valid_rows) == 0:
            raise ValueError("No valid rows found for imputation")
        
        # Use valid data for imputation
        valid_data = X[valid_rows]
        
        # Initialize result tensor on GPU
        result = X.clone()
        
        # Process in batches
        for i in range(0, len(missing_rows), self.batch_size):
            batch_indices = missing_rows[i:i + self.batch_size]
            batch_data = X[batch_indices]
            batch_mask = mask[batch_indices]
            
            # Calculate pairwise distances on GPU
            distances = torch.cdist(batch_data, valid_data)
            
            # Get k nearest neighbors
            _, neighbor_indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
            
            # Compute imputed values for each feature
            for j, (row_idx, row_mask) in enumerate(zip(batch_indices, batch_mask)):
                # Get indices of missing features for this row
                masked_features = torch.nonzero(row_mask, as_tuple=True)[0]
                
                if masked_features.numel() > 0:  # Check if there are any missing features
                    # Get values from k nearest neighbors
                    neighbor_values = valid_data[neighbor_indices[j]]
                    
                    for feature_idx in masked_features:
                        neighbor_feature_values = neighbor_values[:, feature_idx]
                        valid_values = neighbor_feature_values[~torch.isnan(neighbor_feature_values)]
                        
                        if valid_values.numel() > 0:
                            result[row_idx, feature_idx] = torch.mean(valid_values)
                        else:
                            # Fallback to column mean if no valid neighbors
                            col_mean = torch.mean(valid_data[:, feature_idx][~torch.isnan(valid_data[:, feature_idx])])
                            result[row_idx, feature_idx] = col_mean
            
            # Log progress
            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                print(f"Processed {i + self.batch_size}/{len(missing_rows)} rows")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return result.cpu().numpy()

    def _calculate_distances(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise distances efficiently on GPU using batching"""
        n_query = query.size(0)
        n_reference = reference.size(0)
        distances = torch.zeros(n_query, n_reference, device=self.device)
        
        for i in range(0, n_query, self.batch_size):
            batch_query = query[i:i + self.batch_size]
            batch_distances = torch.cdist(batch_query, reference)
            distances[i:i + self.batch_size] = batch_distances
            
        return distances

def parallel_process_features(df: pd.DataFrame, feature_group: List[str]) -> pd.DataFrame:
    """Process a group of features in parallel"""
    processed_df = df[feature_group].copy()
    
    if feature_group[0].startswith('Temperature') or feature_group[0].startswith('Wind'):
        # Use GPU for numerical computations if available
        if torch.cuda.is_available():
            try:
                arr = cp.array(processed_df.values)
                processed_df = pd.DataFrame(
                    cp.asnumpy(arr),
                    columns=processed_df.columns,
                    index=processed_df.index
                )
            except Exception as e:
                print(f"GPU processing failed, falling back to CPU: {e}")
    
    return processed_df



class ProgressTracker:
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

class AccidentDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets) - 1
        self.weights = self._calculate_sample_weights()
    
    def _calculate_sample_weights(self) -> np.ndarray:
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
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_sizes: List[int] = MODEL_CONFIG['model_params']['hidden_sizes'], 
                 dropout_rate: float = MODEL_CONFIG['model_params']['dropout_rate']):
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
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.cuda()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        logits = self.output(x)
        return torch.softmax(logits, dim=1) if not self.training else logits

class AccidentSeverityPredictor:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config['gpu_params']['cuda_devices'][0])
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")
            
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = GPUKNNImputer(n_neighbors=5, batch_size=1000)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Enable memory efficient attention
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    def _extract_temporal_features_gpu(self, timestamps: pd.Series) -> torch.Tensor:
        """Extract temporal features using GPU acceleration"""
        # Convert timestamps to numpy arrays of integers
        hours = timestamps.dt.hour.values
        days = timestamps.dt.dayofweek.values
        months = timestamps.dt.month.values
        
        # Convert to torch tensors and move to GPU
        hours_tensor = torch.tensor(hours, dtype=torch.float32, device=self.device)
        days_tensor = torch.tensor(days, dtype=torch.float32, device=self.device)
        months_tensor = torch.tensor(months, dtype=torch.float32, device=self.device)
        
        # Stack tensors along new dimension
        temporal_features = torch.stack([hours_tensor, days_tensor, months_tensor], dim=1)
        
        return temporal_features

    def _process_temporal_batch_gpu(self, batch_timestamps: torch.Tensor) -> torch.Tensor:
        """Process a batch of temporal data on GPU"""
        # Extract features for the batch
        hours = batch_timestamps[:, 0]
        days = batch_timestamps[:, 1]
        
        # Calculate derived features
        is_weekend = (days >= 5).float()
        is_rush_hour = ((hours >= 7) & (hours <= 9) | (hours >= 16) & (hours <= 18)).float()
        
        return torch.stack([is_weekend, is_rush_hour], dim=1)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing with robust handling of missing values,
        outliers, and edge cases.
        
        Args:
            df (pd.DataFrame): Raw input dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe with selected features
        """
        print("Starting data preprocessing...")
        progress = ProgressTracker(total_steps=7, description="Preprocessing")
        
        df_processed = df.copy()
         # Debugging: Inspect raw dataset
        print("Raw dataset info:")
        print(df_processed.info())
        print("Missing values in raw dataset:")
        print(df_processed.isna().sum())
        # Pre-cleaning step: Replace invalid values across all numeric columns
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        
        try:
            # 1. Process timestamps
            progress.set_description("Processing timestamps")
            timestamp_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
            
            for col in timestamp_columns:
                # Convert to datetime with coercion of invalid values to NaN
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                
                if df_processed[col].notna().any():
                    # Use median of valid timestamps
                    median_timestamp = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_timestamp)
                else:
                    # Fallback to current time if no valid timestamps
                    df_processed[col] = df_processed[col].fillna(pd.Timestamp.now())
            
            # Calculate valid duration
            df_processed['Duration_Minutes'] = (
                df_processed['End_Time'] - df_processed['Start_Time']
            ).dt.total_seconds() / 60
            
            # Filter invalid durations
            duration_mask = (
                (df_processed['Duration_Minutes'] < 0) | 
                (df_processed['Duration_Minutes'] > 24*60*7)  # Max 7 days
            )
            if duration_mask.any():
                median_duration = df_processed.loc[~duration_mask, 'Duration_Minutes'].median()
                df_processed.loc[duration_mask, 'Duration_Minutes'] = median_duration
            
            progress.update()
            
            # 2. Extract temporal features
            progress.set_description("Extracting temporal features")
            
            # Safe extraction of temporal features with validation
            try:
                temporal_features = self._extract_temporal_features_gpu(df_processed['Start_Time'])
                temporal_data = temporal_features.cpu().numpy()
                
                df_processed['Hour'] = temporal_data[:, 0]
                df_processed['Day_Of_Week'] = temporal_data[:, 1]
                df_processed['Month'] = temporal_data[:, 2]
                
            except Exception as e:
                logger.warning(f"GPU temporal feature extraction failed: {e}")
                print("Falling back to CPU processing")
                
                df_processed['Hour'] = df_processed['Start_Time'].dt.hour
                df_processed['Day_Of_Week'] = df_processed['Start_Time'].dt.dayofweek
                df_processed['Month'] = df_processed['Start_Time'].dt.month
            
            # Add derived temporal features
            df_processed['Is_Weekend'] = df_processed['Day_Of_Week'].isin([5, 6]).astype(int)
            df_processed['Is_Rush_Hour'] = df_processed['Hour'].isin([7,8,9,16,17,18]).astype(int)
            night_hours = list(range(22, 24)) + list(range(0, 5))  # Convert ranges to lists before adding

            df_processed['Is_Night'] = df_processed['Hour'].isin(night_hours).astype(int)
            
            progress.update()
            
            # 3. Process weather features
            progress.set_description("Processing weather features")
            weather_features = {
                'Temperature(F)': {'min': -100, 'max': 150, 'default': 70},
                'Wind_Chill(F)': {'min': -100, 'max': 150, 'default': 70},
                'Humidity(%)': {'min': 0, 'max': 100, 'default': 50},
                'Pressure(in)': {'min': 25, 'max': 35, 'default': 29.92},
                'Visibility(mi)': {'min': 0, 'max': 100, 'default': 10},
                'Wind_Speed(mph)': {'min': 0, 'max': 200, 'default': 0},
                'Precipitation(in)': {'min': 0, 'max': 50, 'default': 0}
            }
            
            for feature, limits in weather_features.items():
                # Convert to numeric, coercing errors to NaN
                df_processed[feature] = pd.to_numeric(df_processed[feature], errors='coerce')
                
                # Mark out-of-range values as NaN
                df_processed.loc[
                    (df_processed[feature] < limits['min']) | 
                    (df_processed[feature] > limits['max']),
                    feature
                ] = np.nan
                
                # Handle missing values
                if df_processed[feature].isna().all():
                    df_processed[feature] = limits['default']
                else:
                    median_val = df_processed[feature].median()
                    df_processed[feature] = df_processed[feature].fillna(median_val)
            
            # Add derived weather features
            df_processed['Severe_Weather'] = (
                (df_processed['Temperature(F)'] > 95) |
                (df_processed['Temperature(F)'] < 32) |
                (df_processed['Wind_Speed(mph)'] > 30) |
                (df_processed['Visibility(mi)'] < 1) |
                (df_processed['Precipitation(in)'] > 0.3)
            ).astype(int)
            
            progress.update()
            
            # 4. Process road features
            progress.set_description("Processing road features")
            road_features = [
                'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
                'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
            ]
            
            for feature in road_features:
                # Convert to numeric and fill missing with 0
                df_processed[feature] = pd.to_numeric(
                    df_processed[feature].fillna(0), 
                    errors='coerce'
                ).fillna(0).astype(int)
            
            progress.update()
            
            # 5. Process location features
            progress.set_description("Processing location features")
            
            # Validate coordinates
            coord_columns = ['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng']
            for col in coord_columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Use median for invalid coordinates
                invalid_coords = df_processed[col].isna()
                if invalid_coords.any():
                    median_coord = df_processed[col].median()
                    df_processed.loc[invalid_coords, col] = median_coord
            
            # Calculate distance features
            df_processed['Distance_From_Start_To_End'] = np.sqrt(
                (df_processed['End_Lat'] - df_processed['Start_Lat'])**2 +
                (df_processed['End_Lng'] - df_processed['Start_Lng'])**2
            )
            
            # Handle invalid distances
            distance_mask = (
                df_processed['Distance_From_Start_To_End'].isna() |
                (df_processed['Distance_From_Start_To_End'] < 0) |
                (df_processed['Distance_From_Start_To_End'] > 100)  # Max reasonable distance
            )
            if distance_mask.any():
                median_distance = df_processed.loc[~distance_mask, 'Distance_From_Start_To_End'].median()
                df_processed.loc[distance_mask, 'Distance_From_Start_To_End'] = median_distance
            
            progress.update()
            
            # 6. Process categorical features
            progress.set_description("Processing categorical features")
            
            # Light conditions
            light_conditions = [
                'Sunrise_Sunset', 'Civil_Twilight', 
                'Nautical_Twilight', 'Astronomical_Twilight'
            ]
            
            for condition in light_conditions:
                df_processed[f'Is_{condition}_Night'] = (
                    df_processed[condition].fillna('Day').str.lower() == 'night'
                ).astype(int)
            
            progress.update()
            
            # 7. Final feature selection and scaling
            progress.set_description("Scaling features")
            
            # Select final feature set
            feature_columns = (
                list(weather_features.keys()) +
                road_features +
                ['Duration_Minutes', 'Distance(mi)', 'Distance_From_Start_To_End',
                'Hour', 'Day_Of_Week', 'Month', 'Is_Weekend', 'Is_Rush_Hour',
                'Is_Night', 'Severe_Weather'] +
                [col for col in df_processed.columns if col.startswith('Is_') and 
                col not in ['Is_Weekend', 'Is_Rush_Hour', 'Is_Night']]
            )
            
            # Identify numerical features for scaling
            numerical_features = [
                col for col in feature_columns 
                if df_processed[col].dtype in ['float64', 'int64']
            ]
            
            # Final validation before scaling
            for feature in numerical_features:
                # Replace any remaining invalid values
                df_processed[feature] = df_processed[feature].replace(
                    [np.inf, -np.inf], np.nan
                )
                
                if df_processed[feature].isna().any():
                    median_val = df_processed[feature].median()
                    if pd.isna(median_val):
                        df_processed[feature] = df_processed[feature].fillna(0)
                    else:
                        df_processed[feature] = df_processed[feature].fillna(median_val)
            
            # Scale features with variation
            features_to_scale = [
                feature for feature in numerical_features 
                if df_processed[feature].nunique() > 1
            ]
            
            if features_to_scale:
                scaled_features = self.scaler.fit_transform(
                    df_processed[features_to_scale].values
                )
                df_processed[features_to_scale] = scaled_features
            
            progress.update()
            progress.close()
            
            # Ensure Severity column exists
            if 'Severity' not in df_processed.columns:
                raise ValueError("Severity column missing from input data")
                
            # Validate final feature set
            final_columns = feature_columns + ['Severity']
            missing_columns = set(final_columns) - set(df_processed.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Preprocessing completed in {progress.get_elapsed_time()}")
            
            return df_processed[final_columns]
            
        except Exception as e:
            progress.close()
            logger.error(f"Error during preprocessing: {e}")
            raise Exception(f"Preprocessing failed: {str(e)}") 
    def train(self, df: pd.DataFrame) -> None:
            try:
                # Use smaller chunks for preprocessing
                chunk_size = 100000
                num_chunks = len(df) // chunk_size + 1
                processed_chunks = []

                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(df))
                    chunk = df.iloc[start_idx:end_idx]
                    processed_chunk = self.preprocess_data(chunk)
                    processed_chunks.append(processed_chunk)
                    clear_gpu_cache()                
                processed_data = pd.concat(processed_chunks, axis=0)
                
                X = processed_data.drop('Severity', axis=1).values
                y = processed_data['Severity'].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config['training_params']['test_size'],
                    stratify=y,
                    random_state=self.config['training_params']['random_state']
                )
                
                # Free memory
                del X, y, processed_data, processed_chunks
                clear_gpu_cache()
                
                # Create datasets with memory optimization
                train_dataset = AccidentDataset(X_train, y_train)
                test_dataset = AccidentDataset(X_test, y_test)
                
                # Use memory optimized DataLoader
                train_loader = MemoryOptimizedDataLoader(
                    train_dataset,
                    batch_size=self.config['model_params']['batch_size'],
                    sampler=WeightedRandomSampler(
                        weights=train_dataset.get_sample_weights(),
                        num_samples=len(train_dataset),
                        replacement=True
                    ),
                    pin_memory=self.config['gpu_params']['pin_memory'],
                    num_workers=self.config['gpu_params']['num_workers'],
                    empty_cache_freq=self.config['gpu_params']['empty_cache_freq']
                )
                
                test_loader = MemoryOptimizedDataLoader(
                    test_dataset,
                    batch_size=self.config['model_params']['batch_size'],
                    pin_memory=self.config['gpu_params']['pin_memory'],
                    num_workers=self.config['gpu_params']['num_workers'],
                    empty_cache_freq=self.config['gpu_params']['empty_cache_freq']
                )
                
                # Initialize model with gradient checkpointing
                input_size = X_train.shape[1]
                self.model = AccidentSeverityModel(
                    input_size=input_size,
                    num_classes=len(np.unique(y_train))
                ).to(self.device)
                
                self.model.train()
                torch.cuda.empty_cache()
                
                # Enable gradient checkpointing for memory efficiency
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                
                # Setup training with mixed precision
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.config['model_params']['learning_rate'],
                    weight_decay=self.config['model_params']['weight_decay']
                )
                
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.1,
                    patience=self.config['model_params']['patience']//2
                )
                
                # Train with memory optimization
                self._train_loop(
                    train_loader,
                    test_loader,
                    criterion,
                    optimizer,
                    scheduler
                )
                
            except Exception as e:
                print(f"Error during training: {e}")
                raise
    
    def _train_loop(self, train_loader: DataLoader, 
                    test_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: ReduceLROnPlateau) -> None:
        best_f1 = 0
        epochs_without_improvement = 0
        history = {'train_loss': [], 'val_loss': [], 'f1': []}
        
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
        
        epoch_progress = ProgressTracker(
            total_steps=self.config['model_params']['epochs'],
            description="Training"
        )
        
        for epoch in range(self.config['model_params']['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            batch_progress = ProgressTracker(
                total_steps=len(train_loader),
                description=f"Epoch {epoch+1}/{self.config['model_params']['epochs']}"
            )
            
            for features, targets in train_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = criterion(outputs, targets)
                
                # Scale loss and perform backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['model_params']['grad_clip']
                )
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                batch_progress.update()
            
            batch_progress.close()
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, f1_score = self._validate(test_loader, criterion)
            
            # Update learning rate
            scheduler.step(f1_score)
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['f1'].append(f1_score)
            
            # Update progress description with metrics
            epoch_progress.set_description(
                f"Epoch {epoch+1}/{self.config['model_params']['epochs']} "
                f"- Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - F1: {f1_score:.4f}"
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
            if epochs_without_improvement >= self.config['model_params']['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        epoch_progress.close()
        print(f"Training completed in {epoch_progress.get_elapsed_time()}")
        
        # Load best model and visualize results
        self._load_best_model()
        self._plot_training_history(history)

    @torch.cuda.amp.autocast()
    def _validate(self, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model with GPU optimization and mixed precision"""
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
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
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
        """Save model state and training history"""
        try:
            model_state = self.model.module.state_dict() if isinstance(
                self.model, nn.DataParallel
            ) else self.model.state_dict()
            
            model_info = {
                'model_state': model_state,
                'scaler_state': self.scaler.__dict__,
                'imputer_state': self.imputer.__dict__,
                'history': history,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(model_info, self.model_dir / filename)
            print(f"Model saved successfully: {filename}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    def _load_best_model(self) -> None:
        """Load the best performing model"""
        try:
            model_path = self.model_dir / 'best_accident_model.pth'
            model_info = torch.load(model_path, map_location=self.device)
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_info['model_state'])
            else:
                self.model.load_state_dict(model_info['model_state'])
                
            self.scaler.__dict__.update(model_info['scaler_state'])
            self.imputer.__dict__.update(model_info['imputer_state'])
            print("Best model loaded successfully")
            
        except Exception as e:
            print(f"Error loading best model: {e}")
            raise

    def predict(self, df: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions using GPU acceleration"""
        try:
            print("Starting prediction process...")
            
            # Preprocess input data
            processed_data = self.preprocess_data(df)
            features = processed_data.drop('Severity', axis=1).values
            
            # Create tensor and move to GPU
            features_tensor = torch.FloatTensor(features).to(
                self.device, non_blocking=True
            )
            
            # Make predictions in batches
            batch_size = self.config['model_params']['batch_size'] * 2
            predictions = []
            probabilities = []
            
            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                predict_progress = ProgressTracker(
                    total_steps=(len(features) + batch_size - 1) // batch_size,
                    description="Generating predictions"
                )
                
                for i in range(0, len(features), batch_size):
                    batch = features_tensor[i:i+batch_size]
                    outputs = self.model(batch)
                    
                    if return_proba:
                        probs = torch.softmax(outputs, dim=1)
                        probabilities.append(probs.cpu())
                    else:
                        _, preds = torch.max(outputs.data, 1)
                        predictions.append(preds.cpu())
                    
                    predict_progress.update()
                
                predict_progress.close()
            
            if return_proba:
                return torch.cat(probabilities, dim=0).numpy()
            
            predictions = torch.cat(predictions, dim=0)
            return predictions.numpy() + 1  # Convert back to 1-based indexing
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def _plot_training_history(self, history: Dict) -> None:
        """Plot training metrics history"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 5))
        
        # Plot losses
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

def main():
    """Main function demonstrating usage"""
    try:
        # Initialize progress tracking
        main_progress = ProgressTracker(total_steps=4, description="Overall Progress")
        
        # Load data
        print("Loading US Accident dataset...")
        df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_March23.csv")  # Update path as needed
        main_progress.update()
        
        # Initialize predictor
        predictor = AccidentSeverityPredictor()
        main_progress.update()
        
        # Train model
        print("Starting model training...")
        predictor.train(df)
        main_progress.update()
        
        # Make predictions
        print("Making predictions...")
        predictions = predictor.predict(df)
        
        # Save predictions
        pd.DataFrame({
            'Actual': df['Severity'],
            'Predicted': predictions
        }).to_csv('predictions.csv', index=False)
        
        main_progress.update()
        main_progress.close()
        
        print("Model training and prediction completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()