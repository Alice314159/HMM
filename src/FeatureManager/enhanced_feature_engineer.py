import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any

from src.FeatureManager.feature_calculator import FeatureCalculator
from src.FeatureManager.feature_cleaner import FeatureCleaner
from src.FeatureManager.feature_transfromer import FeatureTransformer
from src.utils.configLoader import HMMConfigReader
from loguru import logger
class EnhancedFeatureEngineer:
    """整合所有功能的主要特征工程类"""

    def __init__(self, config: HMMConfigReader):
        self.config = config
        self.calculator = FeatureCalculator()
        self.cleaner = FeatureCleaner()
        self.transformer = FeatureTransformer(config)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """训练特征工程流水线"""
        logger.info("[EnhancedFeatureEngineer] 开始训练流水线...")

        # 1. 计算特征
        df_features = self.calculator.calculate_all_features(df)


        # 2. 清理特征
        df_clean = self.cleaner.clean_features(df_features)

        df_total = pd.concat([df, df_clean], axis=1)

        # 3. 训练变换器
        self.transformer.fit(df_total)

        self.is_fitted = True

        self.save_pipeline(self.config.output.model_path)
        logger.info("[EnhancedFeatureEngineer] 流水线训练完成")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """变换数据"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        # 1. 计算特征
        features = self.calculator.calculate_all_features(df)
        logger.info("[EnhancedFeatureEngineer] 特征计算完成，特征数量: {}", len(features.columns))

        # 2. 清理特征
        features = self.cleaner.clean_features(features)
        logger.info("[EnhancedFeatureEngineer] 特征清理完成，剩余特征数量: {}", len(features.columns))

        df_total = pd.concat([df, features], axis=1)

        # 3. 应用变换
        return self.transformer.transform(df_total)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """训练并变换数据"""
        self.fit(df)
        return self.transform(df)

    def save_pipeline(self, path: str):
        """保存完整流水线"""
        self.transformer.save_transformer(path)

    def load_pipeline(self, path: str):
        """加载完整流水线"""
        self.transformer.load_transformer(path)
        self.is_fitted = True
        return self

    @property
    def feature_names(self) -> List[str]:
        """获取特征名称"""
        return self.transformer.feature_names