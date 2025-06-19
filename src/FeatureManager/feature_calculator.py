import pandas as pd
import numpy as np
from loguru import logger
from src.utils.basicDefination import ColumnNames, ConfigDefaults, Constants


class FeatureCalculator:
    """专门负责特征计算的类"""

    def __init__(self):
        pass

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        logger.info("开始计算所有特征...")

        if df is None or df.empty:
            raise ValueError("输入数据为空")

        # 验证必要的列是否存在
        self._validate_columns(df)

        try:
            # 计算各类特征
            features_list = []

            # 价格特征
            price_features = self.calculate_price_features(df)
            features_list.append(price_features)
            logger.info(f"价格特征: {price_features.shape}")

            # 技术指标
            technical_features = self.calculate_technical_indicators(df)
            features_list.append(technical_features)
            logger.info(f"技术指标: {technical_features.shape}")

            # 波动率特征
            volatility_features = self.calculate_volatility_features(df)
            features_list.append(volatility_features)
            logger.info(f"波动率特征: {volatility_features.shape}")

            # 趋势指标
            trend_features = self.calculate_trend_indicators(df)
            features_list.append(trend_features)
            logger.info(f"趋势指标: {trend_features.shape}")

            # 价格结构指标
            structure_features = self.calculate_price_structure_indicators(df)
            features_list.append(structure_features)
            logger.info(f"价格结构指标: {structure_features.shape}")

            # 成交量指标
            volume_features = self.calculate_volume_indicators(df)
            features_list.append(volume_features)
            logger.info(f"成交量指标: {volume_features.shape}")

            # 稳定性指标
            stability_features = self.calculate_stability_indicators(df)
            features_list.append(stability_features)
            logger.info(f"稳定性指标: {stability_features.shape}")

            # 动量指标
            momentum_features = self.calculate_momentum_indicators(df)
            features_list.append(momentum_features)
            logger.info(f"动量指标: {momentum_features.shape}")

            # 统计指标
            statistical_features = self.calculate_statistical_indicators(df)
            features_list.append(statistical_features)
            logger.info(f"统计指标: {statistical_features.shape}")

            # 相关性指标
            correlation_features = self.calculate_correlation_indicators(df)
            features_list.append(correlation_features)
            logger.info(f"相关性指标: {correlation_features.shape}")

            # 合并所有特征
            all_features = pd.concat(features_list, axis=1)

            if all_features is None or all_features.empty:
                raise ValueError("特征计算结果为空")

            logger.info(f"所有特征计算完成，形状: {all_features.shape}")
            return all_features

        except Exception as e:
            logger.error(f"特征计算失败: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """验证必要的列是否存在"""
        required_columns = [
            ColumnNames.CLOSE,
            ColumnNames.HIGH,
            ColumnNames.LOW,
            ColumnNames.OPEN,
            ColumnNames.VOLUME
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")

    def _get_column(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """获取指定列的数据"""
        if column_name not in df.columns:
            raise ValueError(f"列 {column_name} 在DataFrame中不存在")
        return df[column_name]

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格相关特征"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')

        # 计算收益率特征
        windows = ColumnNames.RETURNS_WINDOWS
        for window in windows:
            features[ColumnNames.get_returns_column(window, '')] = close.pct_change(periods=window).clip(
                Constants.MIN_RETURN, Constants.MAX_RETURN)
            features[ColumnNames.get_returns_column(window, 'log')] = np.log(close / close.shift(window)).fillna(0)
            features[ColumnNames.get_returns_column(window, 'abs')] = abs(close.pct_change(periods=window).fillna(0))

        return features

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')
        high = self._get_column(df, ColumnNames.HIGH).fillna(method='ffill')
        low = self._get_column(df, ColumnNames.LOW).fillna(method='ffill')

        # Moving averages
        for window in [5, 10, 20, 60]:
            ma = close.rolling(window).mean()
            features[ColumnNames.get_ma_column(window, True)] = close / (ma + Constants.EPSILON)
            features[ColumnNames.get_ma_column(window, False)] = (close - ma) / (ma + Constants.EPSILON)

        # 专门添加需要的MA比率特征
        ma_20 = close.rolling(20).mean()
        ma_60 = close.rolling(60).mean()
        features['ma_ratio_20'] = close / (ma_20 + Constants.EPSILON)
        features['ma_ratio_60'] = close / (ma_60 + Constants.EPSILON)

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features[ColumnNames.MACD_SIGNAL] = signal
        features['macd_hist'] = macd - signal

        # RSI
        for window in [14, 30]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + Constants.EPSILON)
            features[ColumnNames.get_rsi_column(window)] = 100 - (100 / (1 + rs))

        # KDJ指标
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = 100 * (close - low_min) / (high_max - low_min + Constants.EPSILON)
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d

        features['kdj_k'] = k
        features['kdj_d'] = d
        features['kdj_j'] = j

        return features

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE)
        returns = close.pct_change().fillna(0)

        # Rolling volatility
        for window in ColumnNames.VOLATILITY_WINDOWS:
            features[ColumnNames.get_volatility_column(window)] = returns.rolling(window).std()
            features[ColumnNames.get_realized_vol_column(window)] = np.sqrt(
                returns.rolling(window).var() * Constants.TRADING_DAYS_PER_YEAR)

        # 专门添加60日波动率
        features['volatility_60'] = returns.rolling(60).std()

        # GARCH波动率 (简化版本)
        features['garch_vol'] = self._calculate_garch_volatility(returns)

        return features

    def _calculate_garch_volatility(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """计算GARCH波动率的简化版本"""
        # 简化的GARCH(1,1)模型
        alpha = 0.1
        beta = 0.85
        gamma = 0.05

        garch_vol = pd.Series(index=returns.index, dtype=float)

        # 初始化
        initial_vol = returns.rolling(window).std().fillna(returns.std())
        garch_vol.iloc[0] = initial_vol.iloc[0] if not pd.isna(initial_vol.iloc[0]) else 0.01

        for i in range(1, len(returns)):
            if pd.isna(returns.iloc[i - 1]) or pd.isna(garch_vol.iloc[i - 1]):
                garch_vol.iloc[i] = garch_vol.iloc[i - 1] if not pd.isna(garch_vol.iloc[i - 1]) else 0.01
            else:
                garch_vol.iloc[i] = np.sqrt(
                    gamma +
                    alpha * (returns.iloc[i - 1] ** 2) +
                    beta * (garch_vol.iloc[i - 1] ** 2)
                )

        return garch_vol

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')
        high = self._get_column(df, ColumnNames.HIGH).fillna(method='ffill')
        low = self._get_column(df, ColumnNames.LOW).fillna(method='ffill')

        # ADX calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_period = min(14, max(1, len(tr.dropna()) // 2))
        atr = tr.rolling(atr_period, min_periods=1).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)

        plus_dm_smooth = plus_dm_series.rolling(atr_period, min_periods=1).mean()
        minus_dm_smooth = minus_dm_series.rolling(atr_period, min_periods=1).mean()

        atr_safe = atr.copy()
        atr_safe[atr_safe <= 0] = Constants.EPSILON
        atr_safe = atr_safe.fillna(Constants.EPSILON)

        plus_di = 100 * plus_dm_smooth / atr_safe
        minus_di = 100 * minus_dm_smooth / atr_safe

        di_sum = plus_di + minus_di
        di_sum_safe = di_sum.copy()
        di_sum_safe[di_sum_safe <= 0] = Constants.EPSILON

        dx = 100 * abs(plus_di - minus_di) / di_sum_safe
        adx = dx.rolling(atr_period, min_periods=1).mean()

        features['adx'] = adx
        features['di_plus'] = plus_di
        features['di_minus'] = minus_di

        return features

    def calculate_price_structure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格结构指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')
        high = self._get_column(df, ColumnNames.HIGH).fillna(method='ffill')
        low = self._get_column(df, ColumnNames.LOW).fillna(method='ffill')
        open_price = self._get_column(df, ColumnNames.OPEN).fillna(method='ffill')

        # Basic structure indicators
        features['hl_spread'] = (high - low) / (close + Constants.EPSILON)
        features[ColumnNames.OC_SPREAD] = (close - open_price) / (open_price + Constants.EPSILON)

        # Price position indicators
        features['price_range_position'] = (close - low) / (high - low + Constants.EPSILON)

        return features

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        features = pd.DataFrame(index=df.index)

        volume = self._get_column(df, ColumnNames.VOLUME).fillna(method='ffill')

        # Volume statistics
        for window in ColumnNames.VOLUME_WINDOWS:
            vol_std = volume.rolling(window).std()
            vol_mean = volume.rolling(window).mean()
            features[ColumnNames.get_volume_std_column(window)] = vol_std / (vol_mean + Constants.EPSILON)

        return features

    def calculate_stability_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算稳定性指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')

        # Trend stability
        for window in [20, 60]:
            ma = close.rolling(window).mean()
            std = close.rolling(window).std()
            features[f"trend_strength_{window}"] = abs(ma - ma.shift(window)) / (std + Constants.EPSILON)

        return features

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')

        # 动量指标
        features['momentum_20'] = close / close.shift(20) - 1

        # 动量稳定性
        momentum_20 = close.pct_change(20)
        features['momentum_stability_20'] = momentum_20.rolling(20).std() / (
                    abs(momentum_20.rolling(20).mean()) + Constants.EPSILON)

        return features

    def calculate_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算统计指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')

        # Z-Score指标
        close_mean_60 = close.rolling(60).mean()
        close_std_60 = close.rolling(60).std()
        features[ColumnNames.PRICE_ZSCORE_60] = (close - close_mean_60) / (close_std_60 + Constants.EPSILON)

        # 价格Z-Score (使用全历史数据)
        features[ColumnNames.PRICE_ZSCORE_60] = (close - close.expanding().mean()) / (close.expanding().std() + Constants.EPSILON)

        return features

    def calculate_correlation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算相关性指标"""
        features = pd.DataFrame(index=df.index)

        close = self._get_column(df, ColumnNames.CLOSE).fillna(method='ffill')

        # MA斜率比率
        ma_10 = close.rolling(10).mean()
        ma_60 = close.rolling(60).mean()

        ma_slope_10 = ma_10.diff(5) / 5  # 5日斜率
        ma_slope_60 = ma_60.diff(5) / 5  # 5日斜率

        features['ma_slope_ratio_10_60'] = ma_slope_10 / (abs(ma_slope_60) + Constants.EPSILON)

        # 价格与移动平均线的相关性
        features['price_ma_corr_60'] = self._rolling_correlation(close, close.rolling(60).mean(), 60)

        return features

    def _rolling_correlation(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """计算滚动相关系数"""
        corr_values = []

        for i in range(len(x)):
            if i < window - 1:
                corr_values.append(np.nan)
            else:
                start_idx = i - window + 1
                end_idx = i + 1

                x_window = x.iloc[start_idx:end_idx]
                y_window = y.iloc[start_idx:end_idx]

                # 移除NaN值
                valid_mask = ~(pd.isna(x_window) | pd.isna(y_window))
                x_valid = x_window[valid_mask]
                y_valid = y_window[valid_mask]

                if len(x_valid) < 2:
                    corr_values.append(np.nan)
                else:
                    try:
                        corr = np.corrcoef(x_valid, y_valid)[0, 1]
                        corr_values.append(corr if not np.isnan(corr) else 0.0)
                    except:
                        corr_values.append(0.0)

        return pd.Series(corr_values, index=x.index)