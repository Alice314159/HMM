import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils.configLoader import HMMConfigReader
import joblib
from loguru import logger


class FeatureTransformer:
    """专门负责特征变换的类（标准化、特征选择、降维等）"""

    def __init__(self, config: HMMConfigReader):
        if config is None:
            config = {}

        self.config = config

        self.scaler_dict: Dict[str, Any] = {}  # 每列一个scaler
        self.feature_selector = None
        self.pca = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.selected_feature_indices: List[int] = []  # 添加这个来存储选中的特征索引

    def _get_scaler_instance(self):
        """返回新的 scaler 实例"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(self.config.scaler_type, RobustScaler())

    def fit(self, features: pd.DataFrame):
        """训练变换器"""
        logger.info("[FeatureTransformer] 开始训练变换器...")

        # 1. 特征选择
        if self.config.use_manual_features and self.config.manual_features:
            available_features = [col for col in self.config.manual_features if col in features.columns]
            features = features[available_features]
            self.feature_names = available_features
            logger.info(f"使用手动特征选择: {len(available_features)} 个特征")

        elif self.config.top_k_features > 0:
            dummy_target = pd.qcut(features.iloc[:, 0].rank(), q=4, labels=False)
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.config.top_k_features, features.shape[1])
            )
            features_selected = self.feature_selector.fit_transform(features, dummy_target)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_feature_indices = selected_indices.tolist()  # 存储选中的特征索引
            self.feature_names = [features.columns[i] for i in selected_indices]
            features = pd.DataFrame(features_selected, columns=self.feature_names, index=features.index)
            logger.info(f"使用自动特征选择: {len(self.feature_names)} 个特征")

        # 2. 每列单独标准化
        if self.config.scaler_type != 'none':
            self.scaler_dict = {}
            for col in features.columns:
                scaler = self._get_scaler_instance()
                scaler.fit(features[[col]])
                self.scaler_dict[col] = scaler
            logger.info(f"训练独立列标准化器: {self.config.scaler_type}")

        # 3. PCA
        if self.config.pca_variance_ratio and self.config.pca_variance_ratio < 1.0:
            features_scaled = features.copy()
            for col in features_scaled.columns:
                if col in self.scaler_dict:
                    features_scaled[col] = self.scaler_dict[col].transform(features_scaled[[col]])
            self.pca = PCA(n_components=self.config.pca_variance_ratio, random_state=42)
            self.pca.fit(features_scaled.values)
            self.feature_names = [f'PC{i + 1}' for i in range(self.pca.n_components_)]
            logger.info(f"训练PCA: {self.pca.n_components_} 个主成分")

        self.is_fitted = True
        logger.info("[FeatureTransformer] 变换器训练完成")
        return self

    def transform(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """应用变换"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法")

        # 1. 特征选择
        if self.config.use_manual_features and self.config.manual_features:
            available_features = [col for col in self.config.manual_features if col in features.columns]
            df_features = features[available_features]
            logger.info(f"使用手动特征选择: {len(available_features)} 个特征")
        elif self.feature_selector:
            # 修复：使用特征选择器变换数据，然后正确构建DataFrame
            selected_features = self.feature_selector.transform(features)
            # 使用原始特征的列名来构建选中特征的列名
            original_columns = features.columns.tolist()
            selected_column_names = [original_columns[i] for i in self.selected_feature_indices]
            df_features = pd.DataFrame(
                selected_features,
                columns=selected_column_names,  # 使用基于索引重新构建的特征名称
                index=features.index
            )
            # 确保feature_names与实际的列名一致
            self.feature_names = selected_column_names
        else:
            df_features = features[self.feature_names] if self.feature_names else features

        logger.info(f"选择的特征数量: {len(df_features.columns)}")

        # 2. 每列单独标准化
        if self.config.scaler_type != 'none':
            logger.info("应用独立列标准化")
            for col in df_features.columns:
                if col in self.scaler_dict:
                    df_features[col] = self.scaler_dict[col].transform(df_features[[col]]).flatten()

        # 3. PCA
        if self.pca:
            features_array = self.pca.transform(df_features.values)
            logger.info(f"PCA变换后特征数量: {features_array.shape[1]}")
        else:
            features_array = df_features.values
            logger.info(f"未应用PCA，特征数量: {features_array.shape[1]}")

        return features_array, self.feature_names, df_features.index

    def fit_transform(self, features: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
        """训练并变换"""
        self.fit(features)
        return self.transform(features)

    def save_transformer(self, path: str):
        """保存变换器"""
        if not self.is_fitted:
            raise ValueError("变换器必须先训练才能保存")

        joblib.dump({
            'scaler_dict': self.scaler_dict,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'selected_feature_indices': self.selected_feature_indices,  # 添加这个
            'config': {
                'scaler_type': self.config.scaler_type,
                'use_manual_features': self.config.use_manual_features,
                'manual_features': self.config.manual_features,
                'top_k_features': self.config.top_k_features,
                'pca_variance_ratio': self.config.pca_variance_ratio
            }
        }, path)
        logger.info(f"变换器已保存到: {path}")

    def load_transformer(self, path: str):
        """加载变换器"""
        data = joblib.load(path)
        self.scaler_dict = data['scaler_dict']
        self.feature_selector = data['feature_selector']
        self.pca = data['pca']
        self.feature_names = data['feature_names']
        self.selected_feature_indices = data.get('selected_feature_indices', [])  # 加载索引

        config = data['config']
        self.config.scaler_type = config['scaler_type']
        self.config.use_manual_features = config['use_manual_features']
        self.config.manual_features = config['manual_features']
        self.config.top_k_features = config['top_k_features']
        self.config.pca_variance_ratio = config['pca_variance_ratio']

        self.is_fitted = True
        logger.info(f"变换器已从 {path} 加载")
        return self