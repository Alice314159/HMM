import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

warnings.filterwarnings('ignore')
from loguru import logger
from basicDefination import ColumnNames, ConfigDefaults, Constants

class EnhancedFeatureEngineer:
    """Enhanced Feature Engineering for HMM Training/Testing/Validation"""
    
    def __init__(self, config: Dict):
        self.scaler_type = config.get('scaler', ConfigDefaults.SCALER_TYPE)
        self.feature_selection = config.get('feature_selection', ConfigDefaults.FEATURE_SELECTION)
        self.top_k_features = config.get('top_k_features', ConfigDefaults.TOP_K_FEATURES)
        self.pca_variance_ratio = config.get('pca_variance_ratio', ConfigDefaults.PCA_VARIANCE_RATIO)
        self.scaler = self._get_scaler()
        self.feature_selector = None    
        self.pca = None
        self.is_fitted = False
        self.feature_names = []
        
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """验证必要的列是否存在"""
        logger.info(f"当前DataFrame列名: {list(df.columns)}")
        
        required_columns = [
            ColumnNames.CLOSE,
            ColumnNames.HIGH,
            ColumnNames.LOW,
            ColumnNames.OPEN,
            ColumnNames.VOLUME
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            logger.error(f"可用的列: {list(df.columns)}")
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        logger.info("列名验证通过")
    
    def _get_column(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """获取指定列的数据"""
        if column_name not in df.columns:
            raise ValueError(f"列 {column_name} 在DataFrame中不存在")
        return df[column_name]
    
    def _get_scaler(self):
        """Get scaler based on configuration"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(quantile_range=(10.0, 90.0)),
            'minmax': MinMaxScaler(),
            'none': None
        }
        return scalers.get(self.scaler_type, RobustScaler())
    
    # =============================================================================
    # TRAINING PHASE - Fit transformers
    # =============================================================================
    
    def fit(self, df: pd.DataFrame) -> 'EnhancedFeatureEngineer':
        """Fit all transformers on training data"""
        logger.info("Fitting feature engineering pipeline...")
        
        # Validate and map columns
        self._validate_columns(df)
        
        # Calculate features
        features = self._calculate_all_features(df)
        features = self._clean_features(features)
        
        # Feature selection
        if self.feature_selection and self.top_k_features:
            features = self._fit_feature_selector(features)
        
        # Scaling
        if self.scaler is not None:
            self.scaler.fit(features.values)
            scaled_features = self.scaler.transform(features.values)
        else:
            scaled_features = features.values
        
        # PCA
        if self.pca_variance_ratio:
            self.pca = PCA(n_components=self.pca_variance_ratio, random_state=42)
            self.pca.fit(scaled_features)
            self.feature_names = [f'PC{i+1}' for i in range(self.pca.n_components_)]
            logger.info(f"PCA fitted: {scaled_features.shape[1]} -> {self.pca.n_components_} dimensions")
        else:
            self.feature_names = list(features.columns)
        
        self.is_fitted = True
        logger.info("Feature engineering pipeline fitted successfully")
        return self
    
    def _fit_feature_selector(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit feature selector on training data"""
        # Create dummy target for feature selection
        returns = features['returns'] if 'returns' in features.columns else features.iloc[:, 0]
        target = pd.qcut(returns.rank(), q=4, labels=False)
        
        self.feature_selector = SelectKBest(
            score_func=f_classif, 
            k=min(self.top_k_features, len(features.columns))
        )
        selected_features = self.feature_selector.fit_transform(features.values, target)
        selected_feature_names = [features.columns[i] for i in self.feature_selector.get_support(indices=True)]
        
        logger.info(f"Feature selection fitted: {len(features.columns)} -> {len(selected_feature_names)} features")
        return pd.DataFrame(selected_features, columns=selected_feature_names, index=features.index)
    
    # =============================================================================
    # TRANSFORM PHASE - Apply fitted transformers
    # =============================================================================
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """Transform data using fitted transformers (for test/validation/inference)"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        logger.info("Transforming features using fitted pipeline...")
        
        # Calculate features
        features = self._calculate_all_features(df)
        features = self._clean_features(features)
        
        # Apply feature selection
        if self.feature_selector is not None:
            features_array = self.feature_selector.transform(features.values)
            selected_feature_names = [features.columns[i] for i in self.feature_selector.get_support(indices=True)]
            features = pd.DataFrame(features_array, columns=selected_feature_names, index=features.index)
        
        # Apply scaling
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features.values)
        else:
            scaled_features = features.values
        
        # Apply PCA
        if self.pca is not None:
            final_features = self.pca.transform(scaled_features)
        else:
            final_features = scaled_features
        
        # Numerical stability
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"Features transformed: {final_features.shape[0]} samples, {final_features.shape[1]} features")
        return final_features, self.feature_names, features.index
    
    # =============================================================================
    # FIT_TRANSFORM - Combined fit and transform for training data
    # =============================================================================
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """Fit pipeline and transform training data in one step"""
        self.fit(df)
        return self.transform(df)
    
    # =============================================================================
    # FEATURE CALCULATION METHODS
    # =============================================================================
    
    def _calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features"""
        logger.info("开始计算所有特征...")
        
        # 检查输入数据
        if df is None or df.empty:
            raise ValueError("输入数据为空")
        
        logger.info(f"输入数据形状: {df.shape}")
        logger.info(f"输入数据列: {df.columns.tolist()}")
        
        try:
            # 验证必要的列是否存在
            self._validate_columns(df)
            
            # 计算价格特征
            logger.info("计算价格特征...")
            price_features = self._calculate_price_features(df)
            logger.info(f"价格特征形状: {price_features.shape}")
            
            # 计算技术指标
            logger.info("计算技术指标...")
            technical_features = self._calculate_technical_indicators(df)
            logger.info(f"技术指标形状: {technical_features.shape}")
            
            # 计算波动率特征
            logger.info("计算波动率特征...")
            volatility_features = self._calculate_volatility_features(df)
            logger.info(f"波动率特征形状: {volatility_features.shape}")
            
            # 合并所有特征
            logger.info("合并所有特征...")
            all_features = pd.concat([
                price_features,
                technical_features,
                volatility_features
            ], axis=1)
            
            # 检查结果
            if all_features is None or all_features.empty:
                raise ValueError("特征计算结果为空")
            
            logger.info(f"最终特征形状: {all_features.shape}")
            logger.info(f"特征列: {all_features.columns.tolist()}")
            
            return all_features
            
        except Exception as e:
            logger.error(f"特征计算失败: {str(e)}")
            raise
    
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-related features"""
        logger.info("计算价格相关特征...")
        
        features = pd.DataFrame(index=df.index)
        
        try:
            # 获取收盘价
            close = self._get_column(df, ColumnNames.CLOSE)
            if close is None or close.empty:
                raise ValueError("无法获取收盘价数据")
            
            # 填充缺失值
            close = close.fillna(method='ffill')
            
            # 计算收益率特征
            features[ColumnNames.RETURNS] = close.pct_change().clip(Constants.MIN_RETURN, Constants.MAX_RETURN)
            features[ColumnNames.LOG_RETURNS] = np.log(close).diff().clip(Constants.MIN_RETURN, Constants.MAX_RETURN)
            features[ColumnNames.ABS_RETURNS] = features[ColumnNames.RETURNS].abs()
            
            # 检查计算结果
            if features.empty:
                raise ValueError("价格特征计算结果为空")
            
            logger.info(f"价格特征计算完成，形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"价格特征计算失败: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        logger.info("计算技术指标...")
        features = pd.DataFrame(index=df.index)
        
        try:
            close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')
            
            # Moving averages
            for window in [5, 10, 20, 60]:
                ma = close.rolling(window).mean()
                features[ColumnNames.get_ma_column(window, True)] = close / (ma + Constants.EPSILON)
                features[ColumnNames.get_ma_column(window, False)] = (close - ma) / (ma + Constants.EPSILON)
            
            # Momentum indicators
            for period in [5, 10, 20]:
                features[ColumnNames.get_momentum_column(period)] = close.pct_change(periods=period).clip(-1, 1)
                features[ColumnNames.get_roc_column(period)] = ((close - close.shift(period)) / 
                                            (close.shift(period) + Constants.EPSILON)).clip(-1, 1)
            
            # RSI simplified version
            for window in [14, 30]:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / (loss + Constants.EPSILON)
                features[ColumnNames.get_rsi_column(window)] = 100 - (100 / (1 + rs))
            
            logger.info(f"技术指标计算完成，形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"技术指标计算失败: {str(e)}")
            raise
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        logger.info("计算波动率特征...")
        features = pd.DataFrame(index=df.index)
        
        try:
            close = self._get_column(df, ColumnNames.CLOSE)
            returns = close.pct_change().fillna(0)
            
            # Rolling volatility
            for window in ColumnNames.VOLATILITY_WINDOWS:
                features[ColumnNames.get_volatility_column(window)] = returns.rolling(window).std()
                features[ColumnNames.get_realized_vol_column(window)] = np.sqrt(returns.rolling(window).var() * Constants.TRADING_DAYS_PER_YEAR)
            
            # Volatility ratios
            features[ColumnNames.VOL_RATIO] = (features[ColumnNames.get_volatility_column(5)] / 
                                               (features[ColumnNames.get_volatility_column(60)] + Constants.EPSILON))
            
            # GARCH approximation
            features[ColumnNames.GARCH_VOL] = returns.ewm(alpha=0.1).std()
            
            logger.info(f"波动率特征计算完成，形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"波动率特征计算失败: {str(e)}")
            raise
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features"""
        features = pd.DataFrame(index=df.index)
        volume = self._get_column(df, 'volume').fillna(method='ffill')
        close = self._get_column(df, 'close').fillna(method='ffill')
        
        # Volume moving average ratios
        for window in [10, 20]:
            vol_ma = volume.rolling(window).mean()
            features[f'volume_ratio_{window}'] = volume / (vol_ma + 1e-8)
        
        # Price-volume relationship
        returns = close.pct_change()
        features['price_volume_corr'] = returns.rolling(20).corr(volume.pct_change())
        
        # Volume trend
        features['volume_trend'] = volume.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        
        return features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean features"""
        # Remove rows with too many missing values
        features = features.dropna(thresh=int(len(features.columns) * 0.5))
        
        # Fill remaining missing values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant features
        constant_features = [col for col in features.columns if features[col].nunique() <= 1]
        if constant_features:
            logger.warning(f"Removing constant features: {constant_features}")
            features = features.drop(columns=constant_features)
        
        # Outlier handling
        for col in features.columns:
            Q1, Q3 = features[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            features[col] = features[col].clip(lower_bound, upper_bound)
        
        # Ensure numerical stability
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features
    
    # =============================================================================
    # SAVE/LOAD PIPELINE
    # =============================================================================
    
    def save_pipeline(self, filepath: str) -> None:
        """Save fitted pipeline to disk"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'config': {
                'scaler_type': self.scaler_type,
                'feature_selection': self.feature_selection,
                'top_k_features': self.top_k_features,
                'pca_variance_ratio': self.pca_variance_ratio
            }
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> 'EnhancedFeatureEngineer':
        """Load fitted pipeline from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline_data = joblib.load(filepath)
        
        self.scaler = pipeline_data['scaler']
        self.feature_selector = pipeline_data['feature_selector']
        self.pca = pipeline_data['pca']
        self.feature_names = pipeline_data['feature_names']
        
        config = pipeline_data['config']
        self.scaler_type = config['scaler_type']
        self.feature_selection = config['feature_selection']
        self.top_k_features = config['top_k_features']
        self.pca_variance_ratio = config['pca_variance_ratio']
        
        self.is_fitted = True
        logger.info(f"Pipeline loaded from {filepath}")
        return self


# =============================================================================
# USAGE EXAMPLE FOR HMM WORKFLOW
# =============================================================================

class HMMFeatureProcessor:
    """Wrapper class for HMM-specific feature processing workflow"""
    
    def __init__(self, config: Dict):
        self.feature_engineer = EnhancedFeatureEngineer(config)
        self.pipeline_path = config.get('pipeline_path', 'feature_pipeline.pkl')
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data"""
        logger.info("Normalizing data...")
        return self.feature_engineer.scaler.transform(df)
    
    def prepare_training_data(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """Prepare training data - fit and transform"""
        logger.info("Preparing training data...")
        features, feature_names, index = self.feature_engineer.fit_transform(train_df)
        self.feature_engineer.save_pipeline(self.pipeline_path)
        return features, feature_names, index
    
    def prepare_validation_data(self, val_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """Prepare validation data - transform only"""
        logger.info("Preparing validation data...")
        return self.feature_engineer.transform(val_df)
    
    def prepare_test_data(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """Prepare test data - transform only"""
        logger.info("Preparing test data...")
        return self.feature_engineer.transform(test_df)
    
    def prepare_inference_data(self, inference_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """Prepare inference data - load pipeline and transform"""
        logger.info("Preparing inference data...")
        if not self.feature_engineer.is_fitted:
            self.feature_engineer.load_pipeline(self.pipeline_path)
        return self.feature_engineer.transform(inference_df)

