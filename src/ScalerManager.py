import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
import pickle
import os
from loguru import logger
from typing import Union, List, Dict, Optional
import re

class ScalerManager:
    """管理数据标准化和归一化的类，支持智能scaler类型推荐"""
    
    # 定义特征类型和推荐的scaler映射
    FEATURE_SCALER_MAP = {
        # 收益率相关 - 通常近似正态分布，使用StandardScaler
        'returns': 'standard',
        'log_returns': 'standard',
        'abs_returns': 'power',  # 绝对收益率可能偏斜，使用PowerTransformer
        
        # 移动平均比率 - 围绕1波动，使用StandardScaler
        'ma_ratio': 'standard',
        
        # 移动平均距离 - 可能有异常值，使用RobustScaler
        'ma_distance': 'robust',
        
        # 动量指标 - 可能有极值，使用RobustScaler
        'momentum': 'robust',
        'roc': 'robust',  # Rate of Change
        
        # RSI - 有界指标(0-100)，使用MinMaxScaler
        'rsi': 'minmax',
        
        # 波动率相关 - 通常右偏分布，使用PowerTransformer或RobustScaler
        'volatility': 'power',
        'realized_vol': 'power',
        'vol_ratio': 'robust',
        'garch_vol': 'power',
    }
    
    def __init__(self, scaler_type: str = 'auto', feature_names: Optional[List[str]] = None):
        """
        初始化ScalerManager
        
        Args:
            scaler_type: 标准化器类型，可选 'auto', 'standard', 'minmax', 'robust', 'power', 'quantile'
            feature_names: 特征名称列表，用于自动推荐scaler类型
        """
        self.scaler_type = scaler_type
        self.feature_names = feature_names
        self.scalers = {}  # 存储每个特征或整体的scaler
        self.is_fitted = False
        self.auto_recommendations = {}
        
        if scaler_type == 'auto' and feature_names:
            self._auto_recommend_scalers()
        else:
            self.scaler = self._create_scaler(scaler_type)
    
    def _create_scaler(self, scaler_type: str):
        """创建标准化器"""
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(method='yeo-johnson', standardize=True),
            'quantile': QuantileTransformer(output_distribution='normal', random_state=42)
        }
        
        if scaler_type in scaler_map:
            return scaler_map[scaler_type]
        else:
            logger.warning(f"未知的scaler类型 {scaler_type}，使用StandardScaler")
            return StandardScaler()
    
    def _auto_recommend_scalers(self):
        """根据特征名自动推荐scaler类型"""
        if not self.feature_names:
            logger.warning("没有提供特征名称，无法自动推荐scaler类型")
            return
        
        recommendations = {}
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            recommended_scaler = 'standard'  # 默认
            
            # 使用正则表达式和关键词匹配
            for pattern, scaler_type in self.FEATURE_SCALER_MAP.items():
                if pattern in feature_lower:
                    recommended_scaler = scaler_type
                    break
            
            # 特殊处理一些复合特征
            if re.search(r'rsi_\d+', feature_lower):
                recommended_scaler = 'minmax'
            elif re.search(r'volatility_\d+', feature_lower):
                recommended_scaler = 'power'
            elif re.search(r'ma_ratio_\d+', feature_lower):
                recommended_scaler = 'standard'
            elif re.search(r'ma_distance_\d+', feature_lower):
                recommended_scaler = 'robust'
            
            recommendations[feature] = recommended_scaler
        
        self.auto_recommendations = recommendations
        logger.info(f"自动推荐的scaler类型: {recommendations}")
        
        # 为每个特征创建对应的scaler
        for feature, scaler_type in recommendations.items():
            self.scalers[feature] = self._create_scaler(scaler_type)
    
    def get_recommendations(self) -> Dict[str, str]:
        """获取自动推荐的scaler类型"""
        return self.auto_recommendations.copy()
    

    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None):
        """
        拟合标准化器
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
        """
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        if feature_names:
            self.feature_names = feature_names
        
        # 如果是自动模式且有特征名称，重新推荐
        if self.scaler_type == 'auto' and self.feature_names:
            self._auto_recommend_scalers()
        
        if self.scaler_type == 'auto' and self.scalers:
            # 分别拟合每个特征的scaler
            for i, feature in enumerate(self.feature_names):
                if feature in self.scalers:
                    self.scalers[feature].fit(X_array[:, i:i+1])
        else:
            # 使用单一scaler
            self.scaler.fit(X_array)
        
        self.is_fitted = True
        logger.info(f"已完成拟合")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        转换数据
        
        Args:
            X: 特征矩阵
            
        Returns:
            转换后的特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("Scaler尚未拟合，请先调用fit方法")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if self.scaler_type == 'auto' and self.scalers:
            # 分别转换每个特征
            transformed_features = []
            for i, feature in enumerate(self.feature_names):
                if feature in self.scalers:
                    transformed_col = self.scalers[feature].transform(X_array[:, i:i+1])
                    transformed_features.append(transformed_col)
                else:
                    # 如果没有对应的scaler，使用StandardScaler
                    default_scaler = StandardScaler()
                    transformed_col = default_scaler.fit_transform(X_array[:, i:i+1])
                    transformed_features.append(transformed_col)
            
            return np.hstack(transformed_features)
        else:
            return self.scaler.transform(X_array)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        拟合并转换数据
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            
        Returns:
            转换后的特征矩阵
        """
        return self.fit(X, feature_names).transform(X)
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        反向转换数据
        
        Args:
            X: 转换后的特征矩阵
            
        Returns:
            原始特征矩阵
        """
        if not self.is_fitted:
            raise ValueError("Scaler尚未拟合，请先调用fit方法")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if self.scaler_type == 'auto' and self.scalers:
            # 分别反向转换每个特征
            inverse_features = []
            for i, feature in enumerate(self.feature_names):
                if feature in self.scalers:
                    inverse_col = self.scalers[feature].inverse_transform(X_array[:, i:i+1])
                    inverse_features.append(inverse_col)
            
            return np.hstack(inverse_features)
        else:
            return self.scaler.inverse_transform(X_array)
    
    def save(self, save_path: str):
        """
        保存标准化器
        
        Args:
            save_path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("Scaler尚未拟合，无法保存")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        scaler_data = {
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'auto_recommendations': self.auto_recommendations,
            'scalers': self.scalers if self.scaler_type == 'auto' else {'main': self.scaler}
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(scaler_data, f)
        logger.info(f"Scaler已保存到: {save_path}")
    
    def load(self, load_path: str):
        """
        加载标准化器
        
        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到Scaler文件: {load_path}")
            
        with open(load_path, 'rb') as f:
            scaler_data = pickle.load(f)
            
        self.scaler_type = scaler_data['scaler_type']
        self.is_fitted = scaler_data['is_fitted']
        self.feature_names = scaler_data['feature_names']
        self.auto_recommendations = scaler_data.get('auto_recommendations', {})
        
        if self.scaler_type == 'auto':
            self.scalers = scaler_data['scalers']
        else:
            self.scaler = scaler_data['scalers']['main']
            
        logger.info(f"已加载Scaler配置")
    
    def get_scaling_info(self) -> Dict:
        """
        获取标准化信息
        
        Returns:
            包含标准化信息的字典
        """
        if not self.is_fitted:
            raise ValueError("Scaler尚未拟合")
            
        info = {
            'scaler_type': self.scaler_type,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'auto_recommendations': self.auto_recommendations
        }
        
        if self.scaler_type == 'auto' and self.scalers:
            feature_info = {}
            for feature, scaler in self.scalers.items():
                feature_info[feature] = {
                    'scaler_type': type(scaler).__name__,
                    'scaler_params': self._get_scaler_params(scaler)
                }
            info['feature_scalers'] = feature_info
        else:
            info['scaler_params'] = self._get_scaler_params(self.scaler)
            
        return info
    
    def _get_scaler_params(self, scaler) -> Dict:
        """获取scaler的参数信息"""
        params = {}
        
        if isinstance(scaler, StandardScaler):
            if hasattr(scaler, 'mean_'):
                params.update({
                    'mean': scaler.mean_,
                    'scale': scaler.scale_
                })
        elif isinstance(scaler, MinMaxScaler):
            if hasattr(scaler, 'min_'):
                params.update({
                    'min': scaler.min_,
                    'scale': scaler.scale_,
                    'data_min': scaler.data_min_,
                    'data_max': scaler.data_max_
                })
        elif isinstance(scaler, RobustScaler):
            if hasattr(scaler, 'center_'):
                params.update({
                    'center': scaler.center_,
                    'scale': scaler.scale_
                })
        elif isinstance(scaler, PowerTransformer):
            if hasattr(scaler, 'lambdas_'):
                params.update({
                    'lambdas': scaler.lambdas_,
                    'method': scaler.method
                })
        elif isinstance(scaler, QuantileTransformer):
            if hasattr(scaler, 'quantiles_'):
                params.update({
                    'n_quantiles': scaler.n_quantiles_,
                    'output_distribution': scaler.output_distribution
                })
                
        return params
