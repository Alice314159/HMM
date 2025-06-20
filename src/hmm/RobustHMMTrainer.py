import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Tuple, Any, List, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from hmmlearn.hmm import GaussianHMM
from src.utils.basicDefination import ModelMetrics
from src.utils.configLoader import HMMConfigReader
import joblib
from scipy import stats
from dataclasses import dataclass

warnings.filterwarnings('ignore')
from loguru import logger


@dataclass
class ModelCandidate:
    """模型候选者数据类"""
    model: GaussianHMM
    states: np.ndarray
    metrics: ModelMetrics
    config: dict
    composite_score: float


class RobustHMMTrainer:
    """优化的HMM训练器，具有智能模型选择策略"""

    def __init__(self, config: HMMConfigReader):
        train_config = config.get_trainer_config()
        self.max_attempts = train_config.get('max_attempts', 20)
        self.random_state = 42
        self.n_states = train_config.get('n_states', 5)
        self.k_range = train_config.get('k_range', range(5, 41, 5))
        self.state_range = train_config.get('state_range', range(2, 11))
        self.aic_bic_path = config.output.aic_bic_path

        # 新增配置参数
        self.selection_strategy = train_config.get('selection_strategy',
                                                   'composite')  # 'aic', 'bic', 'composite', 'cross_validation'
        self.early_stopping_patience = train_config.get('early_stopping_patience', 5)
        self.min_improvement = train_config.get('min_improvement', 1e-4)
        self.use_cross_validation = train_config.get('use_cross_validation', False)
        self.cv_folds = train_config.get('cv_folds', 3)

        # 权重配置用于复合评分
        self.score_weights = {
            'log_likelihood': 0.3,
            'aic': -0.2,  # 负权重因为越小越好
            'bic': -0.2,  # 负权重因为越小越好
            'silhouette': 0.15,
            'calinski_harabasz': 0.1,
            'stability': 0.05  # 模型稳定性评分
        }

    def train_with_grid_search(self, X: np.ndarray) -> Tuple[GaussianHMM, np.ndarray, ModelMetrics]:
        """使用网格搜索和多种评估策略训练模型"""
        logger.info("开始优化的HMM模型训练...")
        logger.info(f"数据形状: {X.shape}, 选择策略: {self.selection_strategy}")

        # 定义参数网格
        param_grid = self._get_parameter_grid()
        candidates = []

        for state_num in self.state_range:
            logger.info(f"训练 {state_num} 状态的模型...")
            best_candidate = self._train_single_state_model(X, state_num, param_grid)
            if best_candidate:
                candidates.append(best_candidate)

        if not candidates:
            raise RuntimeError("所有模型训练都失败了")

        # 选择最佳模型
        best_model_candidate = self._select_best_model(candidates, X)

        # 可视化结果
        self._visualize_results(candidates)

        logger.info(f"最终选择: {best_model_candidate.model.n_components} 状态模型, "
                    f"复合得分: {best_model_candidate.composite_score:.4f}")

        return (best_model_candidate.model,
                best_model_candidate.states,
                best_model_candidate.metrics)

    def _get_parameter_grid(self) -> List[Dict]:
        """获取参数网格"""
        return [
            {'covariance_type': 'diag', 'n_iter': 100, 'tol': 1e-3, 'algorithm': 'viterbi'},
            {'covariance_type': 'diag', 'n_iter': 200, 'tol': 1e-4, 'algorithm': 'viterbi'},
            {'covariance_type': 'tied', 'n_iter': 150, 'tol': 1e-3, 'algorithm': 'viterbi'},
            {'covariance_type': 'spherical', 'n_iter': 100, 'tol': 1e-3, 'algorithm': 'viterbi'},
            {'covariance_type': 'diag', 'n_iter': 300, 'tol': 1e-5, 'algorithm': 'map'},
        ]

    def _train_single_state_model(self, X: np.ndarray, n_states: int, param_grid: List[Dict]) -> Optional[
        ModelCandidate]:
        """训练单个状态数的最佳模型"""
        best_candidate = None
        best_score = -np.inf
        attempts_without_improvement = 0

        for attempt, params in enumerate(param_grid * (self.max_attempts // len(param_grid) + 1)):
            if attempts_without_improvement >= self.early_stopping_patience:
                logger.info(f"早停: {n_states} 状态模型在 {attempt} 次尝试后停止")
                break

            try:
                candidate = self._train_single_attempt(X, n_states, params, attempt)
                if candidate is None:
                    continue

                # 计算复合得分
                logger.info(f"尝试 {attempt + 1}/{self.max_attempts} - {n_states} 状态模型")
                logger.info(f"配置: {params}")
                logger.info(f"指标: {candidate.metrics}")
                candidate.composite_score = self._calculate_composite_score(candidate.metrics, X)

                # 检查是否有改进
                improvement = candidate.composite_score - best_score
                if improvement > self.min_improvement:
                    best_candidate = candidate
                    best_score = candidate.composite_score
                    attempts_without_improvement = 0
                    logger.info(f"找到更好的 {n_states} 状态模型: 得分={best_score:.4f}")
                else:
                    attempts_without_improvement += 1

            except Exception as e:
                logger.warning(f"训练尝试失败: {e}")
                attempts_without_improvement += 1
                continue

        return best_candidate

    def _train_single_attempt(self, X: np.ndarray, n_states: int, params: dict, attempt: int) -> Optional[
        ModelCandidate]:
        """单次训练尝试"""
        # 数据预处理和稳定化
        X_stable = self._stabilize_data(X, attempt)

        # 创建模型
        model = GaussianHMM(
            n_components=n_states,
            random_state=self.random_state + attempt,
            **params
        )

        # 训练模型
        model.fit(X_stable)
        states = model.predict(X_stable)

        # 验证模型质量
        if not self._validate_model_quality(model, states):
            return None

        # 计算指标
        metrics = self._calculate_metrics(model, X_stable, states)

        return ModelCandidate(
            model=model,
            states=states,
            metrics=metrics,
            config=params,
            composite_score=0.0  # 稍后计算
        )

    def _stabilize_data(self, X: np.ndarray, attempt: int) -> np.ndarray:
        """数据稳定化处理"""
        # 渐进式噪声水平
        noise_levels = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        noise_level = noise_levels[min(attempt, len(noise_levels) - 1)]

        # 添加噪声并处理异常值
        X_stable = X + np.random.normal(0, noise_level, X.shape)
        X_stable = np.nan_to_num(X_stable, nan=0.0, posinf=1e6, neginf=-1e6)

        # 标准化处理（可选）
        if attempt > 10:  # 在后期尝试中使用标准化
            X_stable = (X_stable - np.mean(X_stable, axis=0)) / (np.std(X_stable, axis=0) + 1e-8)

        return X_stable

    def _validate_model_quality(self, model: GaussianHMM, states: np.ndarray) -> bool:
        """验证模型质量"""
        # 检查收敛性
        if not getattr(model, 'monitor_', None) is None:
            if not getattr(model.monitor_, 'converged', True):
                return False

        # 检查状态数量
        unique_states = len(np.unique(states))
        if unique_states < 2:  # 至少需要2个状态
            return False

        # 检查状态分布的合理性
        state_counts = np.bincount(states)
        min_state_proportion = np.min(state_counts) / len(states)
        if min_state_proportion < 0.01:  # 每个状态至少占1%
            return False

        return True

    def _calculate_composite_score(self, metrics: ModelMetrics, X: np.ndarray) -> float:
        """计算复合评分"""
        if self.selection_strategy == 'aic':
            return -metrics.aic  # 负号因为AIC越小越好
        elif self.selection_strategy == 'bic':
            return -metrics.bic  # 负号因为BIC越小越好
        elif self.selection_strategy == 'log_likelihood':
            return metrics.log_likelihood
        elif self.selection_strategy == 'composite':
            # 归一化各个指标
            normalized_metrics = self._normalize_metrics(metrics, X)

            score = 0.0
            for metric, weight in self.score_weights.items():
                if metric in normalized_metrics:
                    score += weight * normalized_metrics[metric]

            return score
        else:
            return metrics.log_likelihood

    def _normalize_metrics(self, metrics: ModelMetrics, X: np.ndarray) -> Dict[str, float]:
        """归一化指标到[0,1]范围"""
        # 这里需要根据实际情况调整归一化方法
        return {
            'log_likelihood': metrics.log_likelihood / len(X),  # 简单归一化
            'aic': 1.0 / (1.0 + abs(metrics.aic)),  # 倒数归一化
            'bic': 1.0 / (1.0 + abs(metrics.bic)),  # 倒数归一化
            'silhouette': (metrics.silhouette_score + 1) / 2,  # 从[-1,1]映射到[0,1]
            'calinski_harabasz': metrics.calinski_harabasz_score / (metrics.calinski_harabasz_score + 1000),
            'stability': 1.0 if metrics.converged else 0.0
        }

    def _select_best_model(self, candidates: List[ModelCandidate], X: np.ndarray) -> ModelCandidate:
        """选择最佳模型"""
        if self.use_cross_validation:
            return self._select_with_cross_validation(candidates, X)
        else:
            # 基于复合得分选择
            return max(candidates, key=lambda c: c.composite_score)

    def _select_with_cross_validation(self, candidates: List[ModelCandidate], X: np.ndarray) -> ModelCandidate:
        """使用交叉验证选择模型"""
        logger.info("使用交叉验证选择最佳模型...")

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        best_candidate = None
        best_cv_score = -np.inf

        for candidate in candidates:
            cv_scores = []

            for train_idx, val_idx in tscv.split(X):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]

                    # 重新训练模型
                    temp_model = GaussianHMM(
                        n_components=candidate.model.n_components,
                        **candidate.config
                    )
                    temp_model.fit(X_train)
                    score = temp_model.score(X_val)
                    cv_scores.append(score)

                except Exception as e:
                    logger.warning(f"交叉验证折叠失败: {e}")
                    continue

            if cv_scores:
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                logger.info(f"{candidate.model.n_components} 状态模型 CV得分: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

                if mean_cv_score > best_cv_score:
                    best_cv_score = mean_cv_score
                    best_candidate = candidate

        return best_candidate or candidates[0]

    def _visualize_results(self, candidates: List[ModelCandidate]):
        """可视化训练结果"""
        if not candidates:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 提取数据
        n_states = [c.model.n_components for c in candidates]
        aics = [c.metrics.aic for c in candidates]
        bics = [c.metrics.bic for c in candidates]
        log_likelihoods = [c.metrics.log_likelihood for c in candidates]
        composite_scores = [c.composite_score for c in candidates]

        # AIC vs 状态数
        axes[0, 0].plot(n_states, aics, 'bo-', label='AIC')
        axes[0, 0].set_xlabel('状态数')
        axes[0, 0].set_ylabel('AIC')
        axes[0, 0].set_title('AIC vs 状态数')
        axes[0, 0].grid(True)

        # BIC vs 状态数
        axes[0, 1].plot(n_states, bics, 'ro-', label='BIC')
        axes[0, 1].set_xlabel('状态数')
        axes[0, 1].set_ylabel('BIC')
        axes[0, 1].set_title('BIC vs 状态数')
        axes[0, 1].grid(True)

        # 对数似然 vs 状态数
        axes[1, 0].plot(n_states, log_likelihoods, 'go-', label='Log Likelihood')
        axes[1, 0].set_xlabel('状态数')
        axes[1, 0].set_ylabel('对数似然')
        axes[1, 0].set_title('对数似然 vs 状态数')
        axes[1, 0].grid(True)

        # 复合得分 vs 状态数
        axes[1, 1].plot(n_states, composite_scores, 'mo-', label='Composite Score')
        axes[1, 1].set_xlabel('状态数')
        axes[1, 1].set_ylabel('复合得分')
        axes[1, 1].set_title('复合得分 vs 状态数')
        axes[1, 1].grid(True)

        # 标记最佳模型
        best_candidate = max(candidates, key=lambda c: c.composite_score)
        best_n_states = best_candidate.model.n_components

        for ax in axes.flat:
            ax.axvline(x=best_n_states, color='red', linestyle='--', alpha=0.7, label='最佳模型')
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.aic_bic_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"结果可视化已保存到 {self.aic_bic_path}")

    def _calculate_metrics(self, model: GaussianHMM, X: np.ndarray, states: np.ndarray) -> ModelMetrics:
        """计算模型评估指标"""
        log_likelihood = model.score(X)
        n_params = self._calculate_n_params(model)
        n_samples = len(X)

        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params

        try:
            silhouette = silhouette_score(X, states) if len(np.unique(states)) > 1 else 0.0
            calinski_harabasz = calinski_harabasz_score(X, states) if len(np.unique(states)) > 1 else 0.0
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

    # 保留原有的方法以保持向后兼容性
    def train(self, X: np.ndarray) -> Tuple[GaussianHMM, np.ndarray, ModelMetrics]:
        """向后兼容的训练方法"""
        return self.train_with_grid_search(X)

    def predict(self, X: np.ndarray, model: GaussianHMM) -> np.ndarray:
        """使用模型进行预测"""
        return model.predict(X)

    def save_model(self, model: GaussianHMM, path: str):
        """保存HMM模型"""
        joblib.dump(model, path)
        logger.info(f"HMM模型已保存到 {path}")

    def load_model(self, path: str) -> GaussianHMM:
        """加载HMM模型"""
        model = joblib.load(path)
        logger.info(f"HMM模型已从 {path} 加载")
        return model