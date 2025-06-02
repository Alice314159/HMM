import numpy as np
import pandas as pd
import pickle
import joblib
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FeatureConsistencyManager:
    """
    特征一致性管理器
    确保训练和预测时特征维度完全一致
    """
    
    def __init__(self):
        self.feature_names_: Optional[List[str]] = None
        self.feature_order_: Optional[Dict[str, int]] = None
        self.scaler_ = None
        self.feature_selector_ = None
        self.pca_ = None
        self.original_feature_count_ = None
        self.processed_feature_count_ = None
        self.processing_steps_ = []
        
    def fit_transform(self, X: np.ndarray, feature_names: List[str], 
                     scaler=None, feature_selector=None, pca=None) -> Tuple[np.ndarray, List[str]]:
        """
        拟合并转换训练数据
        
        Args:
            X: 输入特征矩阵
            feature_names: 特征名称列表
            scaler: 标准化器
            feature_selector: 特征选择器
            pca: PCA降维器
            
        Returns:
            转换后的特征矩阵和特征名称
        """
        logger.info("开始拟合特征一致性管理器...")
        
        # 保存原始特征信息
        self.feature_names_ = feature_names.copy()
        self.feature_order_ = {name: idx for idx, name in enumerate(feature_names)}
        self.original_feature_count_ = len(feature_names)
        
        logger.info(f"原始特征数量: {self.original_feature_count_}")
        logger.info(f"原始特征名称: {self.feature_names_}")
        
        X_processed = X.copy()
        current_feature_names = feature_names.copy()
        
        # 步骤1: 标准化
        if scaler is not None:
            logger.info("应用标准化...")
            X_processed = scaler.fit_transform(X_processed)
            self.scaler_ = scaler
            self.processing_steps_.append("scaler")
            logger.info("✓ 标准化完成")
        
        # 步骤2: 特征选择
        if feature_selector is not None:
            logger.info("应用特征选择...")
            X_processed = feature_selector.fit_transform(X_processed)
            
            # 获取选中的特征
            if hasattr(feature_selector, 'get_support'):
                selected_mask = feature_selector.get_support()
                current_feature_names = [name for name, selected in zip(current_feature_names, selected_mask) if selected]
            elif hasattr(feature_selector, 'get_feature_names_out'):
                current_feature_names = list(feature_selector.get_feature_names_out())
            
            self.feature_selector_ = feature_selector
            self.processing_steps_.append("feature_selector")
            logger.info(f"✓ 特征选择完成，剩余特征数量: {X_processed.shape[1]}")
            logger.info(f"✓ 选中的特征: {current_feature_names}")
        
        # 步骤3: PCA降维
        if pca is not None:
            logger.info("应用PCA降维...")
            X_processed = pca.fit_transform(X_processed)
            current_feature_names = [f'PC{i+1}' for i in range(X_processed.shape[1])]
            self.pca_ = pca
            self.processing_steps_.append("pca")
            logger.info(f"✓ PCA降维完成，主成分数量: {X_processed.shape[1]}")
            logger.info(f"✓ 解释方差比例: {pca.explained_variance_ratio_}")
        
        self.processed_feature_count_ = X_processed.shape[1]
        
        logger.info(f"特征处理完成:")
        logger.info(f"  - 处理步骤: {' -> '.join(self.processing_steps_)}")
        logger.info(f"  - 原始特征数量: {self.original_feature_count_}")
        logger.info(f"  - 最终特征数量: {self.processed_feature_count_}")
        logger.info(f"  - 最终特征名称: {current_feature_names}")
        
        return X_processed, current_feature_names
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        转换预测数据，确保与训练数据维度一致
        
        Args:
            X: 输入特征矩阵
            feature_names: 特征名称列表
            
        Returns:
            转换后的特征矩阵和特征名称
        """
        if self.feature_names_ is None:
            raise ValueError("管理器未拟合，请先调用fit_transform方法")
        
        logger.info("开始转换预测数据...")
        logger.info(f"输入特征数量: {len(feature_names)}")
        logger.info(f"输入特征名称: {feature_names}")
        
        # 关键修复：先进行特征对齐，再应用预处理步骤
        X_aligned, aligned_feature_names = self._align_features(X, feature_names)
        
        # 验证对齐后的维度
        expected_dim = self.original_feature_count_
        actual_dim = X_aligned.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(f"特征对齐后维度不匹配! 期望: {expected_dim}, 实际: {actual_dim}")
        
        logger.info(f"✓ 特征对齐完成: {X_aligned.shape}")
        
        # 现在应用相同的预处理步骤
        X_processed = X_aligned.copy()
        current_feature_names = aligned_feature_names.copy()
        
        # 应用标准化
        if "scaler" in self.processing_steps_:
            logger.info("应用标准化...")
            if self.scaler_ is None:
                raise ValueError("Scaler未拟合")
            X_processed = self.scaler_.transform(X_processed)
            logger.info("✓ 标准化完成")
        
        # 应用特征选择
        if "feature_selector" in self.processing_steps_:
            logger.info("应用特征选择...")
            if self.feature_selector_ is None:
                raise ValueError("Feature selector未拟合")
            X_processed = self.feature_selector_.transform(X_processed)
            
            # 更新特征名称
            if hasattr(self.feature_selector_, 'get_support'):
                selected_mask = self.feature_selector_.get_support()
                current_feature_names = [name for name, selected in zip(current_feature_names, selected_mask) if selected]
            elif hasattr(self.feature_selector_, 'get_feature_names_out'):
                current_feature_names = list(self.feature_selector_.get_feature_names_out())
            
            logger.info(f"✓ 特征选择完成，剩余特征数量: {X_processed.shape[1]}")
        
        # 应用PCA降维
        if "pca" in self.processing_steps_:
            logger.info("应用PCA降维...")
            if self.pca_ is None:
                raise ValueError("PCA未拟合")
            X_processed = self.pca_.transform(X_processed)
            current_feature_names = [f'PC{i+1}' for i in range(X_processed.shape[1])]
            logger.info(f"✓ PCA降维完成，主成分数量: {X_processed.shape[1]}")
        
        # 验证最终维度
        if X_processed.shape[1] != self.processed_feature_count_:
            raise ValueError(f"最终维度不匹配! 期望: {self.processed_feature_count_}, 实际: {X_processed.shape[1]}")
        
        logger.info(f"预测数据转换完成:")
        logger.info(f"  - 最终特征数量: {X_processed.shape[1]}")
        logger.info(f"  - 最终特征名称: {current_feature_names}")
        
        return X_processed, current_feature_names
    
    def _align_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        对齐特征，确保与训练时特征一致
        
        Args:
            X: 输入特征矩阵
            feature_names: 特征名称列表
            
        Returns:
            对齐后的特征矩阵和特征名称
        """
        logger.info("开始特征对齐...")
        
        # 检查特征差异
        train_features = set(self.feature_names_)
        test_features = set(feature_names)
        
        missing_features = train_features - test_features
        extra_features = test_features - train_features
        
        if missing_features:
            logger.warning(f"预测数据缺少以下特征: {missing_features}")
        if extra_features:
            logger.warning(f"预测数据包含额外特征: {extra_features}")
        
        # 创建对齐后的特征矩阵
        X_aligned = np.zeros((X.shape[0], len(self.feature_names_)))
        
        # 创建特征名到索引的映射
        test_feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # 填充已有特征
        for i, train_feature in enumerate(self.feature_names_):
            if train_feature in test_feature_to_idx:
                test_idx = test_feature_to_idx[train_feature]
                X_aligned[:, i] = X[:, test_idx]
                logger.debug(f"✓ 特征对齐: {train_feature} (训练索引{i} <- 测试索引{test_idx})")
            else:
                logger.warning(f"⚠ 缺失特征 {train_feature}, 使用0填充")
                # X_aligned[:, i] 已经是0，无需额外操作
        
        logger.info(f"特征对齐完成: {X_aligned.shape}")
        logger.info(f"对齐后特征顺序: {self.feature_names_}")
        
        return X_aligned, self.feature_names_.copy()
    
    def save(self, filepath: str):
        """保存管理器状态"""
        state = {
            'feature_names_': self.feature_names_,
            'feature_order_': self.feature_order_,
            'original_feature_count_': self.original_feature_count_,
            'processed_feature_count_': self.processed_feature_count_,
            'processing_steps_': self.processing_steps_,
            'scaler_': self.scaler_,
            'feature_selector_': self.feature_selector_,
            'pca_': self.pca_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"特征一致性管理器已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载管理器状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        for key, value in state.items():
            setattr(self, key, value)
        
        logger.info(f"特征一致性管理器已从 {filepath} 加载")
        logger.info(f"  - 原始特征数量: {self.original_feature_count_}")
        logger.info(f"  - 处理后特征数量: {self.processed_feature_count_}")
        logger.info(f"  - 处理步骤: {self.processing_steps_}")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息摘要"""
        return {
            'original_features': self.feature_names_,
            'original_count': self.original_feature_count_,
            'processed_count': self.processed_feature_count_,
            'processing_steps': self.processing_steps_,
            'is_fitted': self.feature_names_ is not None
        }


