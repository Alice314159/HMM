import pandas as pd
import numpy as np
from loguru import logger
from src.utils.basicDefination import ColumnNames, ConfigDefaults, Constants
class FeatureCleaner:
    """专门负责特征清理的类"""

    def __init__(self):
        pass

    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """清理特征数据"""
        logger.info("开始清理特征...")

        try:
            # 1. 记录清理前状态
            logger.info(f"清理前特征形状: {features.shape}")
            missing_before = features.isnull().sum().sum()
            if missing_before > 0:
                logger.warning(f"清理前缺失值总数: {missing_before}")

            # 2. 移除全为NaN的列
            features = features.dropna(axis=1, how='all')

            # 3. 移除常数列
            constant_features = [col for col in features.columns if features[col].nunique() <= 1]
            if constant_features:
                logger.warning(f"移除常数列: {constant_features}")
                features = features.drop(columns=constant_features)

            # 4. 处理无穷值
            features = features.replace([np.inf, -np.inf], np.nan)

            # 5. 填充缺失值
            features = self._fill_missing_values(features)

            # 6. 处理异常值
            features = self._handle_outliers(features)

            # 7. 最终检查
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

            # 8. 验证结果
            missing_after = features.isnull().sum().sum()
            if missing_after > 0:
                raise ValueError(f"清理后仍存在 {missing_after} 个缺失值")

            logger.info(f"特征清理完成，最终形状: {features.shape}")
            return features

        except Exception as e:
            logger.error(f"特征清理失败: {str(e)}")
            raise

    def _fill_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """填充缺失值"""
        for col in features.columns:
            non_null_values = features[col].dropna()
            if len(non_null_values) == 0:
                features[col] = 0
                continue

            median_val = non_null_values.median()
            std_val = non_null_values.std()

            if std_val < Constants.EPSILON:
                features[col] = features[col].fillna(median_val)
            else:
                features[col] = features[col].fillna(method='ffill').fillna(method='bfill').fillna(median_val)

        return features

    def _handle_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        for col in features.columns:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            features[col] = features[col].clip(lower_bound, upper_bound)

        return features