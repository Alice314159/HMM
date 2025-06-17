import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from hmmlearn.hmm import GaussianHMM
from basicDefination import ModelMetrics
warnings.filterwarnings('ignore')
from loguru import logger

class RobustHMMTrainer:
    """鲁棒的HMM训练器"""
    
    def __init__(self, config: Dict):
        self.n_states = config.get('n_states', 6)
        self.max_attempts = 20
        self.random_state = 42
        self.k_range = config.get('k_range', range(5, 41, 5))
        self.state_range = config.get('state_range', range(2, 11))  # 状态数范围

    def predict(self, X: np.ndarray, model: GaussianHMM) -> np.ndarray:
        """使用模型进行预测"""
        return model.predict(X)
    
    def train(self, X: np.ndarray) -> Tuple[GaussianHMM, np.ndarray, ModelMetrics]:
        """训练HMM模型"""
        logger.info(f"开始训练 {self.n_states} 状态的HMM模型...")
        
        best_model = None
        best_states = None
        best_metrics = None
        best_score = -np.inf
        
        history = []
        
        # 渐进式参数配置
        param_configs = [
            {'covariance_type': 'diag', 'n_iter': 100, 'tol': 1e-3},
            {'covariance_type': 'diag', 'n_iter': 200, 'tol': 1e-4},
            {'covariance_type': 'tied', 'n_iter': 150, 'tol': 1e-3},
            {'covariance_type': 'spherical', 'n_iter': 100, 'tol': 1e-3},
            {'covariance_type': 'diag', 'n_iter': 300, 'tol': 1e-5}
        ]
        
        for attempt in range(self.max_attempts):
            try:
                params = param_configs[min(attempt, len(param_configs) - 1)]
                
                # 添加数值稳定性噪声
                noise_level = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6][min(attempt, 4)]
                X_stable = X + np.random.normal(0, noise_level, X.shape)
                X_stable = np.nan_to_num(X_stable, nan=0.0, posinf=1e6, neginf=-1e6)
                
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=params['covariance_type'],
                    n_iter=params['n_iter'],
                    tol=params['tol'],
                    random_state=self.random_state + attempt,
                    algorithm='viterbi'
                )
                
                model.fit(X_stable)
                states = model.predict(X_stable)
                score = model.score(X_stable)
                
                # 计算评估指标
                metrics = self._calculate_metrics(model, X_stable, states)
                
                logger.info(f"尝试 {attempt + 1}: 分数={score:.4f}, 收敛={metrics.converged}, "
                           f"轮廓系数={metrics.silhouette_score:.4f}")
                
                # 选择最佳模型 - 放宽条件
                if (score > best_score and metrics.converged):  # 只要求收敛和分数更高
                    best_model = model
                    best_states = states
                    best_metrics = metrics
                    best_score = score
                
                history.append({'params': params, 'aic': metrics.aic, 'bic': metrics.bic, 'converged': metrics.converged})
                
            except Exception as e:
                logger.warning(f"训练尝试 {attempt + 1} 失败: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("所有训练尝试都失败了")
        
        logger.info("HMM模型训练成功完成")
        
        # 可视化
        df_hist = pd.DataFrame(history)
        plt.plot(df_hist['aic'], label='AIC')
        plt.plot(df_hist['bic'], label='BIC')
        plt.legend()
        plt.savefig('output/aic_bic_history.png')
        
        return best_model, best_states, best_metrics
    
    def _calculate_metrics(self, model: GaussianHMM, X: np.ndarray, states: np.ndarray) -> ModelMetrics:
        """计算模型评估指标"""
        log_likelihood = model.score(X)
        n_params = self._calculate_n_params(model)
        n_samples = len(X)
        
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params
        
        try:
            silhouette = silhouette_score(X, states)
            calinski_harabasz = calinski_harabasz_score(X, states)
        except:
            silhouette = 0.0
            calinski_harabasz = 0.0
        
        converged = getattr(model, 'monitor_', None) is None or getattr(model.monitor_, 'converged', True)
        
        return ModelMetrics(
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            converged=converged,
            n_params=n_params
        )
    
    def _calculate_n_params(self, model: GaussianHMM) -> int:
        """计算模型参数数量"""
        n_states = model.n_components
        n_features = model.n_features
        
        # 转移矩阵参数
        transition_params = n_states * (n_states - 1)
        
        # 发射参数
        if model.covariance_type == 'full':
            emission_params = n_states * n_features + n_states * n_features * (n_features + 1) // 2
        elif model.covariance_type == 'diag':
            emission_params = n_states * n_features * 2
        elif model.covariance_type == 'tied':
            emission_params = n_states * n_features + n_features * (n_features + 1) // 2
        elif model.covariance_type == 'spherical':
            emission_params = n_states * n_features + n_states
        else:
            emission_params = 0
        
        # 初始状态概率
        initial_params = n_states - 1
        
        return transition_params + emission_params + initial_params


