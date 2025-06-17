import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional,Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
from scipy.stats import normaltest
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
from loguru import logger
from basicDefination import ColumnNames, ConfigDefaults, Constants

class EnhancedFeatureEngineer:
    """Enhanced Feature Engineering for HMM Training/Testing/Validation"""
    
    def __init__(self, config: Optional[Dict] = None):
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
            
            # 计算趋势指标
            logger.info("计算趋势指标...")
            trend_features = self._calculate_trend_indicators_debug(df)
            logger.info(f"趋势指标形状: {trend_features.shape}")
            
            # 计算价格结构指标
            logger.info("计算价格结构指标...")
            structure_features = self._calculate_price_structure_indicators(df)
            logger.info(f"价格结构指标形状: {structure_features.shape}")
            
            # 计算成交量指标
            logger.info("计算成交量指标...")
            volume_features = self._calculate_volume_indicators(df)
            logger.info(f"成交量指标形状: {volume_features.shape}")
            
            # 计算稳定性指标
            logger.info("计算稳定性指标...")
            stability_features = self._calculate_stability_indicators(df)
            logger.info(f"稳定性指标形状: {stability_features.shape}")
            
            # 合并所有特征
            logger.info("合并所有特征...")
            all_features = pd.concat([
                price_features,
                technical_features,
                volatility_features,
                trend_features,
                structure_features,
                volume_features,
                stability_features
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
            windows = ColumnNames.RETURNS_WINDOWS
            for window in windows:
                features[ColumnNames.get_returns_column(window,'')] = close.pct_change(periods=window).clip(Constants.MIN_RETURN, Constants.MAX_RETURN)
                features[ColumnNames.get_returns_column(window, 'log')] = np.log(close / close.shift(window)).fillna(0)
                features[ColumnNames.get_returns_column(window, 'abs')] = abs(close.pct_change(periods=window).fillna(0))

            
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
            high = self._get_column(df, ColumnNames.HIGH).fillna(method='ffill')
            low = self._get_column(df, ColumnNames.LOW).fillna(method='ffill')
            
            # Moving averages
            for window in [5, 10, 20, 60]:
                ma = close.rolling(window).mean()
                features[ColumnNames.get_ma_column(window, True)] = close / (ma + Constants.EPSILON)
                features[ColumnNames.get_ma_column(window, False)] = (close - ma) / (ma + Constants.EPSILON)
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            features[ColumnNames.MACD] = macd
            features[ColumnNames.MACD_SIGNAL] = signal
            features[ColumnNames.MACD_HIST] = macd - signal
            
            # KDJ
            low_min = low.rolling(window=9).min()
            high_max = high.rolling(window=9).max()
            rsv = (close - low_min) / (high_max - low_min + Constants.EPSILON) * 100
            features[ColumnNames.KDJ_K] = rsv.ewm(com=2).mean()
            features[ColumnNames.KDJ_D] = features[ColumnNames.KDJ_K].ewm(com=2).mean()
            features[ColumnNames.KDJ_J] = 3 * features[ColumnNames.KDJ_K] - 2 * features[ColumnNames.KDJ_D]
            
            # Bollinger Bands
            for window in ColumnNames.BB_WINDOWS:
                ma = close.rolling(window=window).mean()
                std = close.rolling(window=window).std()
                upper_band = ma + 2 * std
                lower_band = ma - 2 * std
                features[ColumnNames.get_bb_width_column(window)] = (upper_band - lower_band) / ma
            
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

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean features and handle missing values"""
        logger.info("开始清理特征...")
        
        try:
            # 1. 检查并记录缺失值情况
            missing_before = features.isnull().sum()
            if missing_before.any():
                logger.warning(f"清理前缺失值情况:\n{missing_before[missing_before > 0]}")
            
            # 2. 移除全为NaN的列
            features = features.dropna(axis=1, how='all')
            
            # 3. 移除方差为0的列
            constant_features = [col for col in features.columns if features[col].nunique() <= 1]
            if constant_features:
                logger.warning(f"移除常数列: {constant_features}")
                features = features.drop(columns=constant_features)
            
            # 4. 处理无穷值
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # 5. 对每列分别处理缺失值
            for col in features.columns:
                # 获取非空值的统计信息
                non_null_values = features[col].dropna()
                if len(non_null_values) == 0:
                    # 如果列全为空，用0填充
                    features[col] = 0
                    continue
                
                # 计算统计量用于填充
                mean_val = non_null_values.mean()
                median_val = non_null_values.median()
                std_val = non_null_values.std()
                
                # 如果标准差接近0，使用中位数填充
                if std_val < Constants.EPSILON:
                    features[col] = features[col].fillna(median_val)
                else:
                    # 否则使用前向填充，然后后向填充，最后用中位数填充剩余的空值
                    features[col] = features[col].fillna(method='ffill').fillna(method='bfill').fillna(median_val)
            
            # 6. 处理异常值
            for col in features.columns:
                Q1 = features[col].quantile(0.25)
                Q3 = features[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                features[col] = features[col].clip(lower_bound, upper_bound)
            
            # 7. 确保数值稳定性
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)  # 最后的保险措施
            
            # 8. 检查清理后的结果
            missing_after = features.isnull().sum()
            if missing_after.any():
                logger.error(f"清理后仍存在缺失值:\n{missing_after[missing_after > 0]}")
                raise ValueError("特征清理后仍存在缺失值")
            
            # 9. 记录清理结果
            logger.info(f"特征清理完成，最终形状: {features.shape}")
            logger.info(f"特征列: {features.columns.tolist()}")
            
            return features
            
        except Exception as e:
            logger.error(f"特征清理失败: {str(e)}")
            raise

    def _calculate_trend_indicators_debug(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators with debugging for ADX issue"""
        logger.info("计算趋势指标...")
        features = pd.DataFrame(index=df.index)
        
        try:
            close = self._get_column(df, ColumnNames.CLOSE).ffill()
            high = self._get_column(df, ColumnNames.HIGH).ffill()
            low = self._get_column(df, ColumnNames.LOW).ffill()
            
            # Debug: Check data availability
            logger.info(f"Data length: {len(close)}")
            logger.info(f"Close NaN count: {close.isna().sum()}")
            logger.info(f"High NaN count: {high.isna().sum()}")
            logger.info(f"Low NaN count: {low.isna().sum()}")
            
            # ADX calculation with debugging
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Debug TR calculation
            logger.info(f"TR stats: min={tr.min():.4f}, max={tr.max():.4f}, mean={tr.mean():.4f}")
            logger.info(f"TR NaN count: {tr.isna().sum()}")
            
            # Calculate ATR with smaller window if data is limited
            data_length = len(tr.dropna())
            atr_period = min(14, max(1, data_length // 2))  # Adaptive period
            atr = tr.rolling(atr_period, min_periods=1).mean()
            
            # Debug ATR calculation
            logger.info(f"ATR period used: {atr_period}")
            logger.info(f"ATR stats: min={atr.min():.4f}, max={atr.max():.4f}, mean={atr.mean():.4f}")
            logger.info(f"ATR NaN count: {atr.isna().sum()}")
            logger.info(f"ATR zero count: {(atr == 0).sum()}")
            
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Debug DM calculation
            logger.info(f"Plus DM non-zero count: {np.count_nonzero(plus_dm)}")
            logger.info(f"Minus DM non-zero count: {np.count_nonzero(minus_dm)}")
            logger.info(f"Plus DM max: {np.max(plus_dm):.4f}")
            logger.info(f"Minus DM max: {np.max(minus_dm):.4f}")
            
            # Convert to Series for proper alignment
            plus_dm_series = pd.Series(plus_dm, index=df.index)
            minus_dm_series = pd.Series(minus_dm, index=df.index)
            
            # Calculate smoothed DM values
            dm_period = min(14, max(1, data_length // 2))  # Adaptive period
            plus_dm_smooth = plus_dm_series.rolling(dm_period, min_periods=1).mean()
            minus_dm_smooth = minus_dm_series.rolling(dm_period, min_periods=1).mean()
            
            # Debug smoothed DM
            logger.info(f"Plus DM smooth stats: min={plus_dm_smooth.min():.4f}, max={plus_dm_smooth.max():.4f}")
            logger.info(f"Minus DM smooth stats: min={minus_dm_smooth.min():.4f}, max={minus_dm_smooth.max():.4f}")
            logger.info(f"Plus DM smooth NaN count: {plus_dm_smooth.isna().sum()}")
            logger.info(f"Minus DM smooth NaN count: {minus_dm_smooth.isna().sum()}")
            
            # Safe ATR calculation (avoid division by zero)
            atr_safe = atr.copy()
            atr_safe[atr_safe <= 0] = Constants.EPSILON  # Replace zero/negative with small value
            atr_safe = atr_safe.fillna(Constants.EPSILON)  # Replace NaN with small value
            
            # Calculate DI values
            plus_di = 100 * plus_dm_smooth / atr_safe
            minus_di = 100 * minus_dm_smooth / atr_safe
            
            # Debug DI calculation
            logger.info(f"Plus DI stats: min={plus_di.min():.4f}, max={plus_di.max():.4f}")
            logger.info(f"Minus DI stats: min={minus_di.min():.4f}, max={minus_di.max():.4f}")
            logger.info(f"Plus DI NaN count: {plus_di.isna().sum()}")
            logger.info(f"Minus DI NaN count: {minus_di.isna().sum()}")
            
            # Calculate DX
            di_sum = plus_di + minus_di
            di_sum_safe = di_sum.copy()
            di_sum_safe[di_sum_safe <= 0] = Constants.EPSILON
            
            dx = 100 * abs(plus_di - minus_di) / di_sum_safe
            
            # Calculate ADX
            adx_period = min(14, max(1, data_length // 2))
            adx = dx.rolling(adx_period, min_periods=1).mean()
            
            # Debug final values
            logger.info(f"DX stats: min={dx.min():.4f}, max={dx.max():.4f}")
            logger.info(f"ADX stats: min={adx.min():.4f}, max={adx.max():.4f}")
            logger.info(f"ADX NaN count: {adx.isna().sum()}")
            
            # Store results
            features[ColumnNames.ADX] = adx
            features['di_plus'] = plus_di
            features['di_minus'] = minus_di
            
            # Check first few values
            logger.info("First 10 values:")
            logger.info(f"ATR: {atr.head(10).values}")
            logger.info(f"Plus DI: {plus_di.head(10).values}")
            logger.info(f"Minus DI: {minus_di.head(10).values}")
            logger.info(f"ADX: {adx.head(10).values}")
            
            return features
            
        except Exception as e:
            logger.error(f"趋势指标计算失败: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _calculate_price_structure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price structure indicators"""
        logger.info("计算价格结构指标...")
        features = pd.DataFrame(index=df.index)
        
        try:
            close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')
            high = self._get_column(df, ColumnNames.HIGH).fillna(method='ffill')
            low = self._get_column(df, ColumnNames.LOW).fillna(method='ffill')
            open_price = self._get_column(df, ColumnNames.OPEN).fillna(method='ffill')
            
            # High-Low Spread with normalization
            features[ColumnNames.HL_SPREAD] = (high - low) / close
            
            # Open-Close Spread with normalization
            features[ColumnNames.OC_SPREAD] = (close - open_price) / open_price
            
            # OHLC Center with improved calculation
            typical_price = (high + low + close) / 3
            features[ColumnNames.OHLC_CENTER] = (typical_price - open_price) / open_price
            
            # Close Z-score with multiple windows
            for window in [20, 60]:
                ma = close.rolling(window).mean()
                std = close.rolling(window).std()
                features[f"{ColumnNames.CLOSE_ZSCORE}_{window}"] = (close - ma) / std
            
            # Add price structure indicators
            # Body to shadow ratio
            body = abs(close - open_price)
            upper_shadow = high - np.maximum(open_price, close)
            lower_shadow = np.minimum(open_price, close) - low
            features['body_shadow_ratio'] = body / (upper_shadow + lower_shadow + Constants.EPSILON)
            
            # Price range position
            features['price_range_position'] = (close - low) / (high - low + Constants.EPSILON)
            
            # Add price momentum structure
            for window in [5, 10, 20]:
                returns = close.pct_change()
                momentum = returns.rolling(window).mean()
                features[f'price_momentum_{window}'] = momentum
            
            logger.info(f"价格结构指标计算完成，形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"价格结构指标计算失败: {str(e)}")
            raise
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators"""
        logger.info("计算成交量指标...")
        features = pd.DataFrame(index=df.index)
        
        try:
            volume = self._get_column(df, ColumnNames.VOLUME).fillna(method='ffill')
            
            # Volume Standard Deviation
            for window in ColumnNames.VOLUME_WINDOWS:
                vol_std = volume.rolling(window).std()
                vol_mean = volume.rolling(window).mean()
                features[ColumnNames.get_volume_std_column(window)] = vol_std / vol_mean
            
            # Volume Z-score
            for window in ColumnNames.VOLUME_WINDOWS:
                vol_mean = volume.rolling(window).mean()
                vol_std = volume.rolling(window).std()
                features[f"{ColumnNames.VOLUME_ZSCORE}_{window}"] = (volume - vol_mean) / vol_std
            
            logger.info(f"成交量指标计算完成，形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"成交量指标计算失败: {str(e)}")
            raise
    
    def _calculate_stability_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate stability indicators"""
        logger.info("计算稳定性指标...")
        features = pd.DataFrame(index=df.index)
        
        try:
            close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')
            
            # Calculate MA slopes first
            ma_slopes = {}
            for window in ColumnNames.MA_WINDOWS:
                ma = close.rolling(window).mean()
                # Calculate slope using linear regression
                x = np.arange(window)
                slope = ma.rolling(window).apply(
                    lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan
                )
                ma_slopes[window] = slope
            
            # MA Slope Ratio
            for short_window in [5, 10]:
                for long_window in [20, 60]:
                    if short_window < long_window:
                        short_slope = ma_slopes[short_window]
                        long_slope = ma_slopes[long_window]
                        features[f"{ColumnNames.MA_SLOPE_RATIO}_{short_window}_{long_window}"] = short_slope / (long_slope + Constants.EPSILON)
            
            # Price-MA Correlation
            for window in ColumnNames.MA_WINDOWS:
                ma = close.rolling(window).mean()
                corr = close.rolling(window).corr(ma)
                features[ColumnNames.get_price_ma_corr_column(window)] = corr
            
            # Add trend strength indicator
            for window in [20, 60]:
                ma = close.rolling(window).mean()
                std = close.rolling(window).std()
                features[f"trend_strength_{window}"] = abs(ma - ma.shift(window)) / std
            
            # Add price momentum stability
            for window in [5, 10, 20]:
                returns = close.pct_change()
                momentum = returns.rolling(window).mean()
                momentum_std = returns.rolling(window).std()
                features[f"momentum_stability_{window}"] = momentum / (momentum_std + Constants.EPSILON)
            
            logger.info(f"稳定性指标计算完成，形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"稳定性指标计算失败: {str(e)}")
            raise
    
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
    
    def __init__(self,config: dict):
        self.feature_engineer = EnhancedFeatureEngineer(config)
        self.pipeline_path = config.get('pipeline_path', 'feature_pipeline.pkl')

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for a given DataFrame"""
        logger.info("Calculating all features...")
        return self.feature_engineer._calculate_all_features(df)

    def save_pipeline(self, filepath: str) -> None:
        """Save the feature engineering pipeline"""
        logger.info("Saving feature engineering pipeline...")
        if self.feature_engineer.is_fitted:
            self.feature_engineer.save_pipeline(filepath)
        else:
            logger.warning("Feature engineer not fitted, skipping feature processor save")


    def cal_pac(self,info:Dict[str, Any]) -> Dict[str, Any]:
        if self.feature_engineer.pca is not None:
            info['pca_explained_variance'] = self.feature_engineer.pca.explained_variance_ratio_
            info['pca_cumulative_variance'] = np.cumsum(info['pca_explained_variance'])
        return info


