import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Optional, Tuple
import joblib
import os

from sklearn.feature_selection import SelectKBest, f_classif
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
    
        
    def select_best_features(self, X: np.ndarray, feature_names: List[str], k: int) -> Tuple[np.ndarray, List[str]]:
        """使用SelectKBest选择最佳特征"""
        # 确保k不超过特征数量
        k = min(k, len(feature_names))
        
        # 创建虚拟目标变量用于特征选择
        returns = X[:, feature_names.index('returns')] if 'returns' in feature_names else X[:, 0]
        target = pd.qcut(returns, q=4, labels=False)
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, target)
        
        # 获取选中的特征名称
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        logger.info(f"从 {len(feature_names)} 个特征中选择了 {k} 个特征")
        logger.info(f"选中的特征: {selected_features}")
        
        return X_selected, selected_features
    
    def find_optimal_dimensions(self, X: np.ndarray, feature_names: List[str]) -> Tuple[int, np.ndarray, List[str]]:
        """寻找最优特征维度组合"""
        logger.info("开始寻找最优特征维度组合...")
        
        # 确保k_range不超过特征数量
        max_features = len(feature_names)
        k_range = [k for k in self.k_range if k <= max_features]
        if not k_range:
            k_range = [max_features]
        
        best_k = None
        best_X = None
        best_features = None
        best_score = float('inf')  # 使用AIC作为主要评估指标
        best_metrics = None
        
        results = []
        
        for k in k_range:
            logger.info(f"\n尝试特征数量 k = {k}")
            
            # 选择特征
            X_selected, selected_features = self.select_best_features(X, feature_names, k)
            
            # 训练模型
            try:
                model, states, metrics = self.train(X_selected)
                
                # 计算综合得分 (AIC + BIC - 轮廓系数)
                # 轮廓系数越大越好，所以用负号
                score = metrics.aic + metrics.bic - metrics.silhouette_score
                
                results.append({
                    'k': k,
                    'aic': metrics.aic,
                    'bic': metrics.bic,
                    'silhouette': metrics.silhouette_score,
                    'score': score,
                    'features': selected_features
                })
                
                logger.info(f"k={k} 的结果:")
                logger.info(f"AIC: {metrics.aic:.2f}")
                logger.info(f"BIC: {metrics.bic:.2f}")
                logger.info(f"轮廓系数: {metrics.silhouette_score:.4f}")
                logger.info(f"综合得分: {score:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_X = X_selected
                    best_features = selected_features
                    best_metrics = metrics
                    
            except Exception as e:
                logger.warning(f"k={k} 的训练失败: {str(e)}")
                continue
        
        if best_k is None:
            raise RuntimeError("所有特征维度组合都失败了")
        
        # 保存维度选择结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('output/dimension_selection_results.csv', index=False)
        
        # 绘制维度选择结果
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(results_df['k'], results_df['aic'], 'b-', label='AIC')
        plt.plot(results_df['k'], results_df['bic'], 'r-', label='BIC')
        plt.xlabel('特征数量')
        plt.ylabel('值')
        plt.title('AIC和BIC随特征数量的变化')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(results_df['k'], results_df['silhouette'], 'g-')
        plt.xlabel('特征数量')
        plt.ylabel('轮廓系数')
        plt.title('轮廓系数随特征数量的变化')
        
        plt.subplot(2, 2, 3)
        plt.plot(results_df['k'], results_df['score'], 'k-')
        plt.xlabel('特征数量')
        plt.ylabel('综合得分')
        plt.title('综合得分随特征数量的变化')
        
        plt.tight_layout()
        plt.savefig('output/dimension_selection_analysis.png')
        
        logger.info(f"\n最优特征数量: {best_k}")
        logger.info(f"最优特征: {best_features}")
        logger.info(f"最优模型指标:")
        logger.info(f"AIC: {best_metrics.aic:.2f}")
        logger.info(f"BIC: {best_metrics.bic:.2f}")
        logger.info(f"轮廓系数: {best_metrics.silhouette_score:.4f}")
        
        return best_k, best_X, best_features
    def find_optimal_states(self, X: np.ndarray) -> Tuple[int, GaussianHMM, np.ndarray, ModelMetrics]:
        """寻找最优状态数"""
        logger.info("开始寻找最优状态数...")
        
        results = []
        best_model = None
        best_states = None
        best_metrics = None
        best_score = float('inf')
        
        for n_states in self.state_range:
            logger.info(f"\n尝试状态数 n_states = {n_states}")
            
            try:
                # 使用diag协方差类型训练模型
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=200,
                    random_state=self.random_state,
                    algorithm='viterbi'
                )
                
                model.fit(X)
                states = model.predict(X)
                metrics = self._calculate_metrics(model, X, states)
                
                # 计算综合得分 (AIC + BIC - 轮廓系数)
                score = metrics.aic + metrics.bic - metrics.silhouette_score
                
                results.append({
                    'n_states': n_states,
                    'log_likelihood': metrics.log_likelihood,
                    'aic': metrics.aic,
                    'bic': metrics.bic,
                    'silhouette': metrics.silhouette_score,
                    'score': score
                })
                
                logger.info(f"状态数 {n_states} 的结果:")
                logger.info(f"对数似然: {metrics.log_likelihood:.2f}")
                logger.info(f"AIC: {metrics.aic:.2f}")
                logger.info(f"BIC: {metrics.bic:.2f}")
                logger.info(f"轮廓系数: {metrics.silhouette_score:.4f}")
                logger.info(f"综合得分: {score:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_states = states
                    best_metrics = metrics
                
            except Exception as e:
                logger.warning(f"状态数 {n_states} 的训练失败: {str(e)}")
                continue
        
        if best_model is None:
            raise RuntimeError("所有状态数组合都失败了")
        
        # 保存状态数选择结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('output/state_selection_results.csv', index=False)
        
        # 绘制状态数选择结果
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(results_df['n_states'], results_df['log_likelihood'], 'b-', label='对数似然')
        plt.xlabel('状态数')
        plt.ylabel('对数似然')
        plt.title('对数似然随状态数的变化')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(results_df['n_states'], results_df['aic'], 'r-', label='AIC')
        plt.plot(results_df['n_states'], results_df['bic'], 'g-', label='BIC')
        plt.xlabel('状态数')
        plt.ylabel('值')
        plt.title('AIC和BIC随状态数的变化')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(results_df['n_states'], results_df['silhouette'], 'k-')
        plt.xlabel('状态数')
        plt.ylabel('轮廓系数')
        plt.title('轮廓系数随状态数的变化')
        
        plt.subplot(2, 2, 4)
        plt.plot(results_df['n_states'], results_df['score'], 'm-')
        plt.xlabel('状态数')
        plt.ylabel('综合得分')
        plt.title('综合得分随状态数的变化')
        
        plt.tight_layout()
        plt.savefig('output/state_selection_analysis.png')
        
        logger.info(f"\n最优状态数: {best_model.n_components}")
        logger.info(f"最优模型指标:")
        logger.info(f"对数似然: {best_metrics.log_likelihood:.2f}")
        logger.info(f"AIC: {best_metrics.aic:.2f}")
        logger.info(f"BIC: {best_metrics.bic:.2f}")
        logger.info(f"轮廓系数: {best_metrics.silhouette_score:.4f}")
        
        return best_model.n_components, best_model, best_states, best_metrics

    def save_model(self, model: GaussianHMM, scaler: Optional[object], feature_selector: Optional[object], 
                  pca: Optional[object], save_path: str = 'models/hmm_model.joblib'):
        """保存模型和相关组件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'pca': pca
        }
        joblib.dump(model_data, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str = 'models/hmm_model.joblib') -> Tuple[GaussianHMM, Optional[object], Optional[object], Optional[object]]:
        """加载模型和相关组件"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        model_data = joblib.load(load_path)
        logger.info(f"模型已从 {load_path} 加载")
        return (model_data['model'], model_data['scaler'], 
                model_data['feature_selector'], model_data['pca'])
    
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

    def _analyze_states(self, X: np.ndarray, states: np.ndarray):
        """分析每个状态的样本数、年化收益和年化波动"""
        for state_id in range(self.n_states):
            state_mask = states == state_id
            mean_return = np.mean(X[state_mask, 0])
            volatility = np.std(X[state_mask, 0])
            label = self._classify_state(X[state_mask, 0])
            print(f"状态{state_id} 样本数: {np.sum(state_mask)} 年化收益: {mean_return*252:.4f} 年化波动: {volatility*np.sqrt(252):.4f} 标签: {label}")

    def generate_report(self):
        labels = [state.label for state in self.market_states.values()]
        print("所有状态标签分布：", labels)

