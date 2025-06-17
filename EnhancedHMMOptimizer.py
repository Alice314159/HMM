import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import os

class EnhancedHMMOptimizer:
    def __init__(self):
        self.feature_importance = None
        self.selected_features = None
        self.feature_scores = None
        
    def comprehensive_feature_selection(self, X, y, feature_names, n_features=None):
        """
        综合特征选择方法，结合多种特征选择技术
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - feature_names: 特征名称列表
        - n_features: 要选择的特征数量（可选）
        
        返回:
        - 包含特征选择结果的字典
        """
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        
        # 1. F检验（回归）
        f_selector = SelectKBest(f_regression, k='all')
        f_scores = f_selector.fit(X, y).scores_
        
        # 2. 互信息（回归）
        mi_selector = SelectKBest(mutual_info_regression, k='all')
        mi_scores = mi_selector.fit(X, y).scores_
        
        # 3. 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # 4. PCA贡献分析
        pca = PCA()
        pca.fit(X)
        pca_contribution = np.abs(pca.components_).mean(axis=0)
        
        # 5. 相关性分析
        corr_matrix = np.corrcoef(X.T)
        correlation_scores = np.abs(corr_matrix).mean(axis=1)
        
        # 归一化所有分数
        def normalize_scores(scores):
            return (scores - scores.min()) / (scores.max() - scores.min())
        
        f_scores_norm = normalize_scores(f_scores)
        mi_scores_norm = normalize_scores(mi_scores)
        rf_importance_norm = normalize_scores(rf_importance)
        pca_contribution_norm = normalize_scores(pca_contribution)
        correlation_scores_norm = normalize_scores(correlation_scores)
        
        # 组合所有分数
        combined_scores = (f_scores_norm + mi_scores_norm + rf_importance_norm + 
                         pca_contribution_norm + correlation_scores_norm) / 5
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'F_Score': f_scores_norm,
            'MI_Score': mi_scores_norm,
            'RF_Importance': rf_importance_norm,
            'PCA_Contribution': pca_contribution_norm,
            'Correlation_Score': correlation_scores_norm,
            'Combined_Score': combined_scores
        })
        
        # 按组合分数排序
        importance_df = importance_df.sort_values('Combined_Score', ascending=False)
        
        # 确定要选择的特征数量
        if n_features is None:
            n_features = max(int(len(feature_names) * 0.5), 1)  # 默认选择50%的特征
        
        # 选择特征
        selected_features = importance_df['Feature'].head(n_features).tolist()
        
        # 生成特征分析报告
        self._generate_feature_analysis_report(importance_df, selected_features, X, feature_names)
        
        return {
            'selected_features': selected_features,
            'importance_df': importance_df,
            'f_scores': f_scores,
            'mi_scores': mi_scores,
            'rf_importance': rf_importance,
            'pca_contribution': pca_contribution,
            'correlation_scores': correlation_scores
        }
    
    def _generate_feature_analysis_report(self, importance_df, selected_features, X, feature_names):
        """生成特征分析报告"""
        logger.info("生成特征分析报告...")
        
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        
        # 1. 绘制特征重要性条形图
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Combined_Score', y='Feature', 
                   data=importance_df.head(20))
        plt.title('Top 20 Features by Combined Importance Score')
        plt.tight_layout()
        plt.savefig('output/feature_importance.png')
        plt.close()
        
        # 2. 绘制相关性热力图
        plt.figure(figsize=(12, 10))
        corr_matrix = np.corrcoef(X.T)
        sns.heatmap(corr_matrix, 
                   cmap='coolwarm', center=0, annot=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('output/correlation_matrix.png')
        plt.close()
        
        # 3. 绘制PCA贡献率图
        plt.figure(figsize=(10, 6))
        pca = PCA()
        pca.fit(X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/pca_variance.png')
        plt.close()
        
        # 生成文本报告
        report = []
        report.append("="*50)
        report.append("特征选择分析报告")
        report.append("="*50)
        
        # 选中的特征
        report.append("\n选中的特征:")
        for i, feature in enumerate(selected_features, 1):
            report.append(f"{i}. {feature}")
        
        # 特征重要性排名
        report.append("\n特征重要性排名 (Top 10):")
        top_features = importance_df.head(10)
        for _, row in top_features.iterrows():
            report.append(f"{row['Feature']}: {row['Combined_Score']:.4f}")
        
        # 高相关性特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i,j]) > 0.7:  # 相关系数阈值
                    high_corr_pairs.append(
                        f"{feature_names[i]} - {feature_names[j]}: {corr_matrix[i,j]:.3f}"
                    )
        
        if high_corr_pairs:
            report.append("\n高相关性特征对 (|相关系数| > 0.7):")
            for pair in high_corr_pairs:
                report.append(pair)
        
        # 保存报告
        with open('output/feature_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info("特征分析报告已生成")

