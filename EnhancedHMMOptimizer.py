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

    def generate_emission_matrix_report(self, hmm_model, feature_names, output_path='output/emission_matrix_report.html'):
        """
        生成发射矩阵的可视化报告
        
        参数:
        - hmm_model: 训练好的HMM模型
        - feature_names: 特征名称列表
        - output_path: 输出HTML文件的路径
        """
        logger.info("生成发射矩阵可视化报告...")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取发射矩阵和协方差矩阵
        emission_matrix = hmm_model.means_
        covars = hmm_model.covars_
        
        # 检查特征维度是否匹配
        if emission_matrix.shape[1] != len(feature_names):
            logger.warning(f"发射矩阵特征维度 ({emission_matrix.shape[1]}) 与特征名称数量 ({len(feature_names)}) 不匹配")
            # 如果特征名称过多，截取需要的部分
            if len(feature_names) > emission_matrix.shape[1]:
                feature_names = feature_names[:emission_matrix.shape[1]]
            # 如果特征名称不足，添加占位符
            else:
                feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), emission_matrix.shape[1])])
        
        # 创建HTML报告
        html_content = []
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>HMM发射矩阵分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .section { margin-bottom: 30px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #f5f5f5; }
                .heatmap { margin: 20px 0; }
                .feature-info { margin: 10px 0; }
                .state-info { margin: 10px 0; }
                .warning { color: #856404; background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 4px; }
                .covariance-info { margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>HMM发射矩阵分析报告</h1>
        """)
        
        # 添加维度信息
        html_content.append(f'<div class="section">')
        html_content.append(f'<p>状态数量: {emission_matrix.shape[0]}</p>')
        html_content.append(f'<p>特征数量: {emission_matrix.shape[1]}</p>')
        html_content.append('</div>')
        
        # 添加发射矩阵表格
        html_content.append('<div class="section">')
        html_content.append('<h2>发射矩阵（均值）</h2>')
        html_content.append('<table>')
        
        # 表头
        html_content.append('<tr><th>状态</th>')
        for feature in feature_names:
            html_content.append(f'<th>{feature}</th>')
        html_content.append('</tr>')
        
        # 表格内容
        for state_idx, state_emissions in enumerate(emission_matrix):
            html_content.append(f'<tr><td>状态 {state_idx}</td>')
            for emission in state_emissions:
                html_content.append(f'<td>{emission:.4f}</td>')
            html_content.append('</tr>')
        
        html_content.append('</table>')
        
        # 添加协方差矩阵分析
        html_content.append('<div class="section">')
        html_content.append('<h2>协方差矩阵分析</h2>')
        
        for state_idx, state_covar in enumerate(covars):
            html_content.append(f'<div class="state-info">')
            html_content.append(f'<h3>状态 {state_idx} 的协方差矩阵</h3>')
            
            # 计算特征对之间的相关性
            corr_matrix = np.zeros_like(state_covar)
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    if state_covar[i,i] > 0 and state_covar[j,j] > 0:
                        corr_matrix[i,j] = state_covar[i,j] / np.sqrt(state_covar[i,i] * state_covar[j,j])
            
            # 添加协方差表格
            html_content.append('<table>')
            html_content.append('<tr><th>特征</th>')
            for feature in feature_names:
                html_content.append(f'<th>{feature}</th>')
            html_content.append('</tr>')
            
            for i, feature in enumerate(feature_names):
                html_content.append(f'<tr><td>{feature}</td>')
                for j in range(len(feature_names)):
                    html_content.append(f'<td>{state_covar[i,j]:.4f}</td>')
                html_content.append('</tr>')
            
            html_content.append('</table>')
            
            # 添加相关性分析
            html_content.append('<div class="covariance-info">')
            html_content.append('<h4>特征相关性分析</h4>')
            
            # 找出强相关的特征对
            strong_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if abs(corr_matrix[i,j]) > 0.7:  # 相关系数阈值
                        strong_corr_pairs.append(
                            f"{feature_names[i]} - {feature_names[j]}: {corr_matrix[i,j]:.3f}"
                        )
            
            if strong_corr_pairs:
                html_content.append('<p>强相关特征对 (|相关系数| > 0.7):</p>')
                html_content.append('<ul>')
                for pair in strong_corr_pairs:
                    html_content.append(f'<li>{pair}</li>')
                html_content.append('</ul>')
            else:
                html_content.append('<p>没有发现强相关的特征对</p>')
            
            # 添加方差分析
            variances = np.diag(state_covar)
            max_var_idx = np.argmax(variances)
            min_var_idx = np.argmin(variances)
            
            html_content.append('<h4>方差分析</h4>')
            html_content.append(f'<p>最大方差特征: {feature_names[max_var_idx]} ({variances[max_var_idx]:.4f})</p>')
            html_content.append(f'<p>最小方差特征: {feature_names[min_var_idx]} ({variances[min_var_idx]:.4f})</p>')
            
            html_content.append('</div>')
            html_content.append('</div>')
        
        html_content.append('</div>')
        
        # 添加状态分析
        html_content.append('<div class="section">')
        html_content.append('<h2>状态特征分析</h2>')
        
        for state_idx, state_emissions in enumerate(emission_matrix):
            html_content.append(f'<div class="state-info">')
            html_content.append(f'<h3>状态 {state_idx} 的特征分布</h3>')
            
            # 找出该状态下最重要的特征
            sorted_indices = np.argsort(np.abs(state_emissions))[::-1]
            top_features = [(feature_names[i], state_emissions[i]) for i in sorted_indices[:5]]
            
            html_content.append('<ul>')
            for feature, value in top_features:
                html_content.append(f'<li>{feature}: {value:.4f}</li>')
            html_content.append('</ul>')
            html_content.append('</div>')
        
        html_content.append('</div>')
        
        # 添加特征分析
        html_content.append('<div class="section">')
        html_content.append('<h2>特征状态分析</h2>')
        
        for feature_idx, feature_name in enumerate(feature_names):
            html_content.append(f'<div class="feature-info">')
            html_content.append(f'<h3>{feature_name}</h3>')
            
            # 计算该特征在不同状态下的变化
            feature_values = emission_matrix[:, feature_idx]
            mean_value = np.mean(feature_values)
            std_value = np.std(feature_values)
            
            html_content.append(f'<p>平均值: {mean_value:.4f}</p>')
            html_content.append(f'<p>标准差: {std_value:.4f}</p>')
            
            # 找出该特征最显著的状态
            max_state = np.argmax(np.abs(feature_values))
            html_content.append(f'<p>最显著状态: 状态 {max_state} (值: {feature_values[max_state]:.4f})</p>')
            
            html_content.append('</div>')
        
        html_content.append('</div>')
        
        # 结束HTML
        html_content.append("""
            </div>
        </body>
        </html>
        """)
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        logger.info(f"发射矩阵报告已保存到: {output_path}")
        
        return output_path 