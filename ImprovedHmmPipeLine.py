import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from EnhancedFeatureEngineer import EnhancedFeatureEngineer, HMMFeatureProcessor
from RobustHMMTrainer import RobustHMMTrainer
from loguru import logger
from ScalerManager import ScalerManager

class ImprovedHMMPipeline:
    """Improved HMM Training Pipeline following standard ML workflow"""
    
    def __init__(self, config_path: str):
        # Load configuration file
        if isinstance(config_path, str):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path
        
        # Initialize feature processor
        self.feature_processor = HMMFeatureProcessor(self.config)

        # Initialize scaler manager
        scaler_config = self.config.get('scaler', {})
        if isinstance(scaler_config, dict):
            scaler_type = scaler_config.get('type', 'standard')
        else:
            scaler_type = 'standard'
        self.scaler_manager = ScalerManager(scaler_type=scaler_type)
        
        # Initialize PCA
        pca_config = self.config.get('pca', {})
        self.pca = None
        self.pca_variance_ratio = pca_config.get('variance_ratio', None)
        self.pca_n_components = pca_config.get('n_components', None)
        
        # Initialize other components
        self.hmm_trainer = None
        self.hmm_model = None
        
        # Record processed feature information
        self.feature_names = None
        self.train_index = None
        self.test_index = None
        
        # Training state records
        self.train_states = None
        self.train_metrics = None
        
        # Pipeline state
        self.is_fitted = False

    def split_data_by_time(self, df: pd.DataFrame, cutoff_date: str = '2025-01-01') -> tuple:
        """Split data by time into training and testing sets"""
        logger.info("Splitting data by time...")

        cutoff1 = pd.Timestamp('2017-01-01')
        cutoff = pd.Timestamp(cutoff_date)
        test_cutoff = pd.Timestamp(cutoff_date) - pd.Timedelta(days=45)
        
        train_df = df[(df.index < cutoff) & (df.index >= cutoff1)].copy()
        test_df = df[df.index >= test_cutoff].copy()
        
        logger.info(f"Training data: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
        logger.info(f"Testing data: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")
        
        return train_df, test_df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from raw data"""
        logger.info("Computing features...")
        
        # Check input data
        if df is None or df.empty:
            raise ValueError("Input data is empty")
        
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input data columns: {df.columns.tolist()}")
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Compute features
        try:
            df_features = self.feature_processor.calculate_all_features(df)
            
            # Check feature calculation results
            if df_features is None or df_features.empty:
                raise ValueError("Feature calculation result is empty")
            
            logger.info(f"Computed features shape: {df_features.shape}")
            logger.info(f"Feature columns: {df_features.columns.tolist()}")
            
            # Check for required features
            required_features = ['Return', 'Log_Return', 'Price_Zscore']
            missing_features = [feat for feat in required_features if feat not in df_features.columns]
            if missing_features:
                logger.warning(f"Missing required features: {missing_features}")
                # Calculate missing features
                if 'Return' not in df_features.columns:
                    df_features['Return'] = df['Close'].pct_change()
                if 'Log_Return' not in df_features.columns:
                    df_features['Log_Return'] = np.log(df['Close']).diff()
                if 'Price_Zscore' not in df_features.columns:
                    mean = df['Close'].rolling(window=20).mean()
                    std = df['Close'].rolling(window=20).std()
                    df_features['Price_Zscore'] = (df['Close'] - mean) / (std + 1e-8)

                    # Merge original raw fields and new features
                full_features = pd.concat([df, df_features], axis=1)

                # Handle missing values in full feature set
                missing_values = full_features.isnull().sum()
                if missing_values.any():
                    logger.warning(f"Missing values in features:\n{missing_values[missing_values > 0]}")

                    full_features = full_features.fillna(method='ffill')
                    full_features = full_features.fillna(method='bfill')
                    full_features = full_features.fillna(full_features.median())
                    full_features = full_features.fillna(0)

                    remaining_missing = full_features.isnull().sum().sum()
                    if remaining_missing > 0:
                        logger.warning(f"Still have {remaining_missing} missing values after filling")
                    else:
                        logger.info("Successfully filled all missing values")
                else:
                    logger.info("No missing values detected")

                return full_features

        except Exception as e:
            logger.error(f"Feature calculation failed: {str(e)}")
            raise
    
    def normalize_data(self, df: pd.DataFrame) -> tuple:
        """Normalize data using fitted scaler"""
        logger.info("Normalizing data...")
        
        # 1. 验证输入数据
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        logger.info(f"Input data shape: {df.shape}")
        
        # 2. 检查数据是否包含任何有效值
        if df.size == 0:
            raise ValueError("Input data contains no values")
        
        # 3. 检查是否所有列都是数值型
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Found non-numeric columns: {non_numeric_cols}")
            df = df.select_dtypes(include=['number'])
        
        # 4. 获取特征名称
        feature_names = list(df.columns)
        if not feature_names:
            raise ValueError("No valid feature columns found")
        
        logger.info(f"Feature names: {feature_names}")
        
        # 5. 检查数据中是否存在 NaN 或无穷值
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df.values).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Found {nan_count} NaN values and {inf_count} infinite values")
            # 使用前向填充处理 NaN
            df = df.fillna(method='ffill')
            # 使用后向填充处理剩余的 NaN
            df = df.fillna(method='bfill')
            # 使用 0 填充任何剩余的 NaN
            df = df.fillna(0)
            # 处理无穷值
            df = df.replace([np.inf, -np.inf], 0)
        
        # 6. 验证数据是否仍然为空
        if df.empty:
            raise ValueError("Data is empty after cleaning")
        
        # 7. 获取数据值并验证
        X = df.values
        if X.size == 0:
            raise ValueError("Data array is empty after conversion")
        
        logger.info(f"Data shape before normalization: {X.shape}")
        
        try:
            # 8. 应用标准化
            X_normalized = self.scaler_manager.fit_transform(X, feature_names)
            
            # 9. 验证标准化结果
            if X_normalized.size == 0:
                raise ValueError("Normalized data is empty")
            
            logger.info(f"Data normalized: {X_normalized.shape[0]} samples, {X_normalized.shape[1]} features")
            
            # 10. 保存 scaler
            os.makedirs('models', exist_ok=True)
            self.scaler_manager.save('models/scaler.pkl')
            
            return X_normalized, df.index
            
        except Exception as e:
            logger.error(f"Data normalization failed: {str(e)}")
            raise
    
    def apply_pca(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Apply PCA transformation to the data
        
        Args:
            X: Input data array
            is_training: Whether this is training data (True) or test/validation/inference data (False)
        
        Returns:
            Transformed data array
        """
        logger.info("Applying PCA transformation...")
        logger.info(f"Input data shape: {X.shape}")
        
        # 1. 检查并处理 NaN 值
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in input data")
            
            # 计算每列的统计量
            col_means = np.nanmean(X, axis=0)
            col_medians = np.nanmedian(X, axis=0)
            col_stds = np.nanstd(X, axis=0)
            
            # 对每列分别处理
            for col in range(X.shape[1]):
                # 获取当前列的 NaN 位置
                nan_mask = np.isnan(X[:, col])
                if nan_mask.any():
                    # 如果标准差接近0，使用中位数填充
                    if col_stds[col] < 1e-8:
                        X[nan_mask, col] = col_medians[col]
                    else:
                        # 否则使用均值填充
                        X[nan_mask, col] = col_means[col]
            
            logger.info("Replaced NaN values with appropriate statistics")
        
        # 2. 检查并处理无穷值
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in input data")
            
            # 对每列分别处理
            for col in range(X.shape[1]):
                # 获取当前列的无穷值位置
                inf_mask = np.isinf(X[:, col])
                if inf_mask.any():
                    # 使用该列的最大/最小值替换无穷值
                    col_max = np.nanmax(X[~inf_mask, col])
                    col_min = np.nanmin(X[~inf_mask, col])
                    X[inf_mask & (X[:, col] > 0), col] = col_max
                    X[inf_mask & (X[:, col] < 0), col] = col_min
            
            logger.info("Replaced infinite values with column max/min")
        
        # 3. 再次检查并处理任何剩余的 NaN 或无穷值
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Found remaining NaN or infinite values after initial cleaning")
            
            # 使用更保守的方法处理
            for col in range(X.shape[1]):
                # 获取非 NaN 和非无穷值的索引
                valid_mask = ~(np.isnan(X[:, col]) | np.isinf(X[:, col]))
                if valid_mask.any():
                    # 使用有效值的统计量
                    valid_values = X[valid_mask, col]
                    col_mean = np.mean(valid_values)
                    col_std = np.std(valid_values)
                    
                    # 替换无效值
                    invalid_mask = ~valid_mask
                    if col_std < 1e-8:
                        X[invalid_mask, col] = col_mean
                    else:
                        # 使用均值加上一个小的随机扰动
                        X[invalid_mask, col] = col_mean + np.random.normal(0, col_std * 0.1, size=invalid_mask.sum())
                else:
                    # 如果列全为无效值，使用0填充
                    X[:, col] = 0
            
            logger.info("Applied conservative cleaning for remaining invalid values")
        
        # 4. 最终检查
        if np.isnan(X).any() or np.isinf(X).any():
            logger.error("Data still contains invalid values after all cleaning attempts")
            logger.error(f"NaN count: {np.isnan(X).sum()}")
            logger.error(f"Inf count: {np.isinf(X).sum()}")
            raise ValueError("Data still contains NaN or infinite values after cleaning")
        
        # 5. 应用 PCA
        if is_training:
            # For training data, fit and transform
            if self.pca is None:
                # Initialize PCA
                if self.pca_variance_ratio is not None:
                    self.pca = PCA(n_components=self.pca_variance_ratio, random_state=42)
                elif self.pca_n_components is not None:
                    self.pca = PCA(n_components=self.pca_n_components, random_state=42)
                else:
                    # Default to 95% variance ratio if no configuration provided
                    self.pca = PCA(n_components=0.95, random_state=42)
                
                # Fit and transform
                X_pca = self.pca.fit_transform(X)
                logger.info(f"PCA fitted: {X.shape[1]} -> {self.pca.n_components_} dimensions")
                logger.info(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
                
                # Update feature names
                self.feature_names = [f'PC{i+1}' for i in range(self.pca.n_components_)]
                logger.info(f"PCA feature names: {self.feature_names}")
            else:
                # If PCA is already fitted, just transform
                logger.info(f"Using fitted PCA with {self.pca.n_components_} components")
                logger.info(f"Current feature names: {self.feature_names}")
                X_pca = self.pca.transform(X)
        else:
            # For test/validation/inference data, only transform
            if self.pca is None:
                raise ValueError("PCA must be fitted before transforming test/validation/inference data")
            logger.info(f"Transforming test data with fitted PCA ({self.pca.n_components_} components)")
            logger.info(f"Current feature names: {self.feature_names}")
            X_pca = self.pca.transform(X)
            logger.info(f"Data transformed using fitted PCA: {X.shape[1]} -> {self.pca.n_components_} dimensions")
        
        return X_pca


    
    def train_hmm_model(self, X_train: np.ndarray) -> None:
        """Train HMM model on normalized training data"""
        logger.info("Training HMM model...")
        
        self.hmm_trainer = RobustHMMTrainer(self.config['trainer'])
        
        # Train HMM model
        self.hmm_model, self.train_states, self.train_metrics = self.hmm_trainer.train(X_train)
        
        # Mark pipeline as fitted
        self.is_fitted = True
        
        logger.info("HMM model training completed")
        logger.info(f"Model state count: {self.hmm_model.n_components}")
        logger.info(f"Training data state distribution: {np.bincount(self.train_states)}")
        logger.info(f"Training metrics: {self.train_metrics}")
    

    
    def predict_states(self, X_test: np.ndarray) -> np.ndarray:
        """Predict states for test data"""
        logger.info("Predicting states...")
        
        if self.hmm_model is None:
            raise ValueError("HMM model not trained yet. Call train_hmm_model first.")
        
        # Predict test data states
        test_states = self.hmm_model.predict(X_test)
        
        logger.info("State prediction completed")
        logger.info(f"Test data state distribution: {np.bincount(test_states)}")
        
        return test_states

    
    def save_pipeline(self, save_path: str) -> None:
        """Save complete processing pipeline"""
        logger.info(f"Saving pipeline to: {save_path}")
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving. Call prepare_training_data first.")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save feature processor
        feature_processor_path = os.path.join(save_path, 'feature_processor.pkl')
        self.feature_processor.save_pipeline(feature_processor_path)
        
        # Save scaler
        scaler_path = os.path.join(save_path, 'scaler.pkl')
        self.scaler_manager.save(scaler_path)
        
        # Save PCA
        if self.pca is not None:
            pca_path = os.path.join(save_path, 'pca.pkl')
            with open(pca_path, 'wb') as f:
                pickle.dump(self.pca, f)
        
        # Save pipeline metadata
        pipeline_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'train_states': self.train_states,
            'train_metrics': self.train_metrics,
            'is_fitted': self.is_fitted,
            'train_index': self.train_index,
            'test_index': self.test_index,
            'pca_variance_ratio': self.pca_variance_ratio,
            'pca_n_components': self.pca_n_components
        }
        
        # Save main pipeline data
        with open(os.path.join(save_path, 'pipeline_data.pkl'), 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        # Save HMM model
        if self.hmm_model is not None:
            with open(os.path.join(save_path, 'hmm_model.pkl'), 'wb') as f:
                pickle.dump(self.hmm_model, f)
        
        # Save HMM trainer
        if self.hmm_trainer is not None:
            with open(os.path.join(save_path, 'hmm_trainer.pkl'), 'wb') as f:
                pickle.dump(self.hmm_trainer, f)
        
        logger.info("✓ Pipeline saved successfully")
    
    def load_pipeline(self, load_path: str) -> None:
        """Load complete processing pipeline"""
        logger.info(f"Loading pipeline from {load_path}")
        
        # Load feature processor
        feature_processor_path = os.path.join(load_path, 'feature_processor.pkl')
        if os.path.exists(feature_processor_path):
            self.feature_engineer.load_pipeline(feature_processor_path)
        else:
            logger.warning("Feature processor file not found")
        
        # Load scaler
        scaler_path = os.path.join(load_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler_manager.load(scaler_path)
        else:
            logger.warning("Scaler file not found")
        
        # Load PCA
        pca_path = os.path.join(load_path, 'pca.pkl')
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
        else:
            logger.warning("PCA file not found")
        
        # Load main pipeline data
        pipeline_data_path = os.path.join(load_path, 'pipeline_data.pkl')
        if os.path.exists(pipeline_data_path):
            with open(pipeline_data_path, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            self.config = pipeline_data['config']
            self.feature_names = pipeline_data['feature_names']
            self.train_states = pipeline_data['train_states']
            self.train_metrics = pipeline_data.get('train_metrics')
            self.is_fitted = pipeline_data.get('is_fitted', False)
            self.train_index = pipeline_data.get('train_index')
            self.test_index = pipeline_data.get('test_index')
            self.pca_variance_ratio = pipeline_data.get('pca_variance_ratio')
            self.pca_n_components = pipeline_data.get('pca_n_components')
        
        # Load HMM model
        hmm_model_path = os.path.join(load_path, 'hmm_model.pkl')
        if os.path.exists(hmm_model_path):
            with open(hmm_model_path, 'rb') as f:
                self.hmm_model = pickle.load(f)
        
        # Load HMM trainer
        hmm_trainer_path = os.path.join(load_path, 'hmm_trainer.pkl')
        if os.path.exists(hmm_trainer_path):
            with open(hmm_trainer_path, 'rb') as f:
                self.hmm_trainer = pickle.load(f)
        
        logger.info("✓ Pipeline loaded successfully")
    
    def validate_pipeline_consistency(self, X_train: np.ndarray, X_test: np.ndarray) -> dict:
        """Validate consistency between training and test data processing"""
        logger.info("Validating data processing consistency...")
        
        validation_result = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'shape_match': X_train.shape[1] == X_test.shape[1],
            'feature_count_match': len(self.feature_names) if self.feature_names else 0,
            'feature_names_match': True  # Default to True
        }
        
        # Check feature names consistency
        if self.feature_names:
            validation_result['feature_names_match'] = (
                len(self.feature_names) == X_train.shape[1] and 
                len(self.feature_names) == X_test.shape[1]
            )
        
        # Statistical characteristics check
        if validation_result['shape_match']:
            train_mean = np.mean(X_train, axis=0)
            test_mean = np.mean(X_test, axis=0)
            train_std = np.std(X_train, axis=0)
            test_std = np.std(X_test, axis=0)
            
            mean_diff = np.abs(train_mean - test_mean)
            std_ratio = test_std / (train_std + 1e-8)
            
            validation_result['statistics_check'] = {
                'mean_difference': mean_diff,
                'std_ratio': std_ratio,
                'reasonable_mean_diff': np.all(mean_diff < 2.0),
                'reasonable_std_ratio': np.all((std_ratio > 0.5) & (std_ratio < 2.0))
            }
            
            logger.info(f"Shape validation: {'✓' if validation_result['shape_match'] else '✗'}")
            logger.info(f"Feature names validation: {'✓' if validation_result['feature_names_match'] else '✗'}")
            if 'statistics_check' in validation_result:
                stats_check = validation_result['statistics_check']
                logger.info(f"Mean difference validation: {'✓' if stats_check['reasonable_mean_diff'] else '✗'}")
                logger.info(f"Std ratio validation: {'✓' if stats_check['reasonable_std_ratio'] else '✗'}")
        
        return validation_result
    
    def get_feature_importance(self) -> dict:
        """Get feature importance information"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        importance_info = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_states': self.hmm_model.n_components if self.hmm_model else 0,
            'train_state_distribution': np.bincount(self.train_states) if self.train_states is not None else None
        }
        importance_info = self.feature_processor.cal_pac(importance_info)
        return importance_info

