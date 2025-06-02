import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import  List,  Tuple

import os
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as sp

from hmmlearn.hmm import GaussianHMM
from basicDefination import MarketState
from loguru import logger
warnings.filterwarnings('ignore')

from ImprovedHmmPipeLine import ImprovedHMMPipeline

class IntelligentStateAnalyzer:
    """
    智能状态分析器
    
    推荐用法：
    1. matplotlib图片只保存不显示（不弹窗、不阻塞、不show），所有plot_*方法均自动保存图片并关闭figure。
    2. 推荐使用 save_full_plotly_html_report 生成交互式HTML报告，避免嵌入matplotlib静态图片。
    3. save_html_report 仅用于需要静态图片嵌入HTML的场景，通常不推荐。
    """
    
    # 统一的状态颜色映射，参考用户示例
    STATE_COLOR_MAP = {
        '深度熊市': '#006400',
        '熊市': '#1E90FF',
        '熊市末期': '#4682B4',
        '高波动': '#FFD700',
        '震荡': '#FFD700',
        '平衡震荡': '#FFFF00',
        '温和调整': '#90EE90',
        '缓慢复苏': '#FFA500',
        '稳健上涨': '#FFA07A',
        '牛市': '#FF8C00',
        '牛市加速': '#FF4500',
        '超级牛市': '#B22222',
        '寂牛行情': '#8B0000',
        '强牛市': '#FF8C00',
        '其它': '#CCCCCC'
    }
    
    def __init__(self, model: GaussianHMM, states: np.ndarray, X: np.ndarray, 
                 feature_names: List[str], index: pd.Index, df: pd.DataFrame):
        self.model = model
        self.states = states
        self.X = X
        self.feature_names = feature_names
        self.index = index
        self.df = df
        self.n_states = model.n_components
        self.market_states = {}
        
        self._analyze_states()
    
    def _max_drawdown(self, series):
        running_max = np.maximum.accumulate(series)
        drawdown = (series - running_max) / running_max
        return np.min(drawdown)

    def _analyze_states(self):
        """分析市场状态"""
        # 检查是否有returns特征
        if 'returns' in self.feature_names:
            returns_idx = self.feature_names.index('returns')
            all_returns = self.X[:, returns_idx]
        else:
            # 尝试从原始数据中获取Close价格
            try:
                close_prices = self.df.loc[self.index, 'Close']
                all_returns = close_prices.pct_change().fillna(0).values
            except KeyError:
                # 如果找不到Close列，使用第一个特征作为替代
                logger.warning("未找到'Close'列，使用第一个特征作为替代")
                all_returns = self.X[:, 0]
        
        # 确保所有可能的状态都被分析
        unique_states = np.unique(self.states)
        for state_id in range(self.n_states):
            state_mask = self.states == state_id
            if not np.any(state_mask):
                # 如果状态不存在，使用模型参数创建默认状态
                self.market_states[state_id] = MarketState(
                    state_id=state_id,
                    label=f"状态{state_id}",
                    description="样本不足，使用模型参数",
                    mean_return=float(self.model.means_[state_id][0]),
                    volatility=float(np.sqrt(self.model.covars_[state_id][0, 0])),
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    probability=float(self.model.startprob_[state_id]),
                    avg_duration=0.0,
                    risk_level="未知"
                )
                continue
                
            state_returns = all_returns[state_mask]
            
            # 限制极端收益
            state_returns = np.clip(state_returns, -0.1, 0.1)
            
            # 计算基本统计量
            mean_return = np.mean(state_returns)
            volatility = np.std(state_returns)
            
            # 计算年化指标
            annual_return = mean_return * 252  # 简单年化
            annual_vol = volatility * np.sqrt(252)
            
            # 计算夏普比率
            risk_free_rate = 0.03  # 假设无风险利率为3%
            sharpe_ratio = (annual_return - risk_free_rate) / (annual_vol + 1e-8)
            
            # 计算最大回撤
            indices = np.where(state_mask)[0]
            segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            mdds = []
            for seg in segments:
                if len(seg) < 2:
                    continue
                seg_returns = all_returns[seg]
                safe_returns = np.clip(seg_returns, -0.1, 0.1)  # 限制极端收益
                cum = np.cumprod(1 + safe_returns)
                mdds.append(self._max_drawdown(cum))
            max_drawdown = np.mean(mdds) if mdds else 0
            
            # 其他统计量
            durations = self._get_state_durations(state_id)
            avg_duration = np.mean(durations) if durations else 0
            probability = np.sum(state_mask) / len(self.states)
            
            # 状态分类
            label, description, risk_level = self._classify_state(
                state_returns, mean_return, volatility, sharpe_ratio, max_drawdown
            )
            
            self.market_states[state_id] = MarketState(
                state_id=state_id,
                label=label,
                description=description,
                mean_return=mean_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                probability=probability,
                avg_duration=avg_duration,
                risk_level=risk_level
            )
    
    def _classify_state(self, state_returns: np.ndarray, mean_return: float, volatility: float, 
                       sharpe_ratio: float, max_drawdown: float) -> Tuple[str, str, str]:
        """智能状态分类（适配高波动数据）"""
        annual_return = mean_return * 252
        annual_vol = volatility * np.sqrt(252)

        # 打印调试信息
        logger.info(f"\n状态分类详情:")
        logger.info(f"年化收益率: {annual_return:.4f}")
        logger.info(f"年化波动率: {annual_vol:.4f}")
        logger.info(f"夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"最大回撤: {max_drawdown:.4f}")

        # 强牛市
        if annual_return > 0.15 and annual_vol < 1.0 and sharpe_ratio > 1.5:
            logger.info("分类为: 强牛市")
            return "强牛市", "高收益低波动的理想环境", "低风险"
        # 牛市
        elif annual_return > 0.08 and sharpe_ratio > 0.5:
            logger.info("分类为: 牛市")
            return "牛市", "正收益且风险调整后表现良好", "中风险"
        # 震荡市
        elif abs(annual_return) <= 0.08 and annual_vol < 1.5:
            logger.info("分类为: 震荡市")
            return "震荡市", "收益率接近0，波动适中", "中低风险"
        # 高波动
        elif annual_vol >= 1.5:
            logger.info("分类为: 高波动")
            return "高波动", "波动极大，方向不明", "高风险"
        # 熊市
        elif annual_return < -0.08 or max_drawdown < -0.20:
            logger.info("分类为: 熊市")
            return "熊市", "负收益或大回撤", "高风险"
        # 其它
        else:
            logger.info("分类为: 其它")
            return "其它", "未能归入其他类别", "中性"
    
    def _get_state_durations(self, state_id: int) -> List[int]:
        """获取状态持续时间"""
        durations = []
        current_duration = 0
        
        for s in self.states:
            if s == state_id:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations
    
    def generate_report(self) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append("HMM 市场状态分析报告")
        report.append("=" * 80)
        report.append(f"分析期间: {self.index.min().strftime('%Y-%m-%d')} 至 {self.index.max().strftime('%Y-%m-%d')}")
        report.append(f"总样本数: {len(self.states)}")
        report.append(f"状态数量: {self.n_states}")
        report.append(f"特征数量: {len(self.feature_names)}")
        report.append("")
        
        report.append("各状态详细分析:")
        report.append("-" * 60)
        
        for state in self.market_states.values():
            annual_return = (1 + state.mean_return) ** 252 - 1
            report.append(f"状态 {state.state_id}: {state.label}")
            report.append(f"  描述: {state.description}")
            report.append(f"  出现概率: {state.probability:.2%}")
            report.append(f"  平均收益率: {state.mean_return:.4f} ({annual_return:.2%} 年化)")
            report.append(f"  波动率: {state.volatility:.4f} ({state.volatility*np.sqrt(252):.2%} 年化)")
            report.append(f"  夏普比率: {state.sharpe_ratio:.3f}")
            report.append(f"  最大回撤: {state.max_drawdown:.2%}")
            report.append(f"  平均持续天数: {state.avg_duration:.1f}")
            report.append(f"  风险等级: {state.risk_level}")
            # 年化收益率极端时加警告
            if abs(annual_return) > 2:
                report.append('  ⚠️ 年化收益率极端，可能样本过少或数据异常')
            report.append("")
        
        return "\n".join(report)
    
    def plot_comprehensive_analysis(self):
        """综合可视化分析"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 状态时间序列
        self._plot_state_timeline(axes[0, 0])
        
        # 2. 转移矩阵热图
        self._plot_transition_matrix(axes[0, 1])
        
        # 3. 状态统计对比
        self._plot_state_statistics(axes[0, 2])
        
        # 4. 收益率分布
        self._plot_returns_distribution(axes[1, 0])
        
        # 5. 状态持续时间
        self._plot_state_durations(axes[1, 1])
        
        # 6. 风险收益散点图
        self._plot_risk_return_scatter(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('output/comprehensive_analysis.png')
    
    def _plot_state_timeline(self, ax):
        """绘制状态时间序列"""
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_states))
        state_series = pd.Series(self.states, index=self.index)
        close_prices = self.df.loc[self.index, 'Close']
        
        # 确保所有状态都有对应的颜色和标签
        for state_id in range(self.n_states):
            mask = state_series == state_id
            if not np.any(mask):
                continue
            state_dates = state_series.index[mask]
            state_prices = close_prices[mask]
            state_label = self.market_states[state_id].label if state_id in self.market_states else f"状态{state_id}"
            ax.scatter(state_dates, state_prices, c=[colors[state_id]],
                      label=state_label, alpha=0.7, s=20)
        
        ax.set_title('市场状态时间序列')
        ax.set_xlabel('日期')
        ax.set_ylabel('收盘价')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_transition_matrix(self, ax):
        """绘制转移矩阵"""
        sns.heatmap(self.model.transmat_, annot=True, fmt='.3f', cmap='Blues', 
                   ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('状态转移矩阵')
        ax.set_xlabel('转移到状态')
        ax.set_ylabel('来自状态')
    
    def _plot_state_statistics(self, ax):
        states = list(self.market_states.keys())
        returns = [self.market_states[s].mean_return * 252 for s in states]
        vols = [self.market_states[s].volatility * np.sqrt(252) for s in states]
        bars = ax.bar(range(len(states)), returns, alpha=0.7, color='skyblue', label='年化收益率')
        ax2 = ax.twinx()
        line = ax2.plot(range(len(states)), vols, 'ro-', label='年化波动率')
        ax.set_title('各状态年化收益率与波动率对比')
        ax.set_xlabel('市场状态')
        ax.set_ylabel('年化收益率', color='blue')
        ax2.set_ylabel('年化波动率', color='red')
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([self.market_states[s].label for s in states])
        ax.grid(True, alpha=0.3)
    
    def _plot_returns_distribution(self, ax):
        """绘制收益率分布"""
        returns = [self.market_states[s].mean_return * 252 for s in self.market_states.keys()]
        ax.hist(returns, bins=30, alpha=0.7, color='skyblue', label='年化收益率')
        ax.set_title('各状态年化收益率分布')
        ax.set_xlabel('年化收益率')
        ax.legend()
    
    def _plot_state_durations(self, ax):
        durations = [self.market_states[s].avg_duration for s in self.market_states.keys()]
        labels = [self.market_states[s].label for s in self.market_states.keys()]
        ax.bar(range(len(self.market_states)), durations, alpha=0.7, color='skyblue', label='平均持续天数')
        ax.set_title('各状态平均持续天数')
        ax.set_xlabel('市场状态')
        ax.set_ylabel('平均持续天数')
        ax.set_xticks(range(len(self.market_states)))
        ax.set_xticklabels(labels)
        ax.legend()
    
    def _plot_risk_return_scatter(self, ax):
        returns = [self.market_states[s].mean_return * 252 for s in self.market_states.keys()]
        vols = [self.market_states[s].volatility * np.sqrt(252) for s in self.market_states.keys()]
        scatter_colors = plt.cm.Set3(np.linspace(0, 1, self.n_states))
        scatter = ax.scatter(vols, returns, alpha=0.7, c=scatter_colors, s=100)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=scatter_colors[i],
                                    label=self.market_states[i].label,
                                    markersize=10) for i in range(self.n_states)]
        ax.legend(handles=legend_elements)
        ax.set_title('风险收益散点图')
        ax.set_xlabel('年化波动率')
        ax.set_ylabel('年化收益率')
        ax.grid(True, alpha=0.3)

    def save_all_figures(self, prefix='figure_'):
        """
        保存所有matplotlib分析图片到指定前缀路径下。
        注意：本方法只保存图片，不显示，不弹窗。
        推荐只在需要静态图片时调用。
        """
        # 自动创建目录
        output_dir = os.path.dirname(prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.plot_state_timeline(f'{prefix}timeline.png')
        self.plot_transition_matrix(f'{prefix}transmat.png')
        self.plot_state_statistics(f'{prefix}statistics.png')
        self.plot_returns_distribution(f'{prefix}returns.png')
        self.plot_state_durations(f'{prefix}durations.png')
        self.plot_risk_return_scatter(f'{prefix}risk_return.png')

    def plot_state_timeline(self, filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_state_timeline(ax)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
        plt.savefig('output/state_timeline.png')

    def plot_transition_matrix(self, filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_transition_matrix(ax)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.savefig('output/transition_matrix.png')

    def plot_state_statistics(self, filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_state_statistics(ax)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.savefig('output/state_statistics.png')

    def plot_returns_distribution(self, filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_returns_distribution(ax)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.savefig('output/returns_distribution.png')

    def plot_state_durations(self, filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_state_durations(ax)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.savefig('output/state_durations.png')

    def plot_risk_return_scatter(self, filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_risk_return_scatter(ax)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close(fig)
        else:
            plt.savefig('output/risk_return_scatter.png')

    def _get_unified_state_order(self):
        """统一状态排序逻辑"""
        return sorted(self.market_states.keys())

    def _get_unified_colors(self):
        """统一颜色方案，优先用STATE_COLOR_MAP"""
        state_order = self._get_unified_state_order()
        colors_list = []
        for state_id in state_order:
            label = self.market_states[state_id].label
            colors_list.append(self.STATE_COLOR_MAP.get(label, '#888888'))
        return self.STATE_COLOR_MAP, colors_list

    def _plot_state_timeline_unified(self, ax=None, plotly=False):
        state_order = self._get_unified_state_order()
        color_map, colors_list = self._get_unified_colors()
        if plotly:
            df_plot = pd.DataFrame({
                'date': self.index,
                'close': self.df.loc[self.index, 'Close'].values,
                'state': [self.market_states[int(s)].label for s in self.states]
            })
            traces = []
            for i, state_id in enumerate(state_order):
                if state_id not in self.market_states:
                    continue
                label = self.market_states[state_id].label
                sub = df_plot[df_plot['state'] == label]
                if len(sub) == 0:
                    continue
                traces.append(go.Scatter(
                    x=sub['date'], y=sub['close'],
                    mode='markers',
                    name=label,
                    marker=dict(size=6, color=color_map.get(label, '#888888')),
                    hovertemplate=f'日期: %{{x}}<br>收盘价: %{{y}}<br>状态: {label}'
                ))
            fig = go.Figure(traces)
            fig.update_layout(title="市场状态时间序列", xaxis_title="日期", yaxis_title="收盘价")
            return fig
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            state_series = pd.Series(self.states, index=self.index)
            close_prices = self.df.loc[self.index, 'Close']
            for i, state_id in enumerate(state_order):
                if state_id not in self.market_states:
                    continue
                mask = state_series == state_id
                if not np.any(mask):
                    continue
                state_dates = state_series.index[mask]
                state_prices = close_prices[mask]
                label = self.market_states[state_id].label
                ax.scatter(state_dates, state_prices, 
                          c=colors_list[i], label=label, 
                          alpha=0.7, s=20)
            ax.set_title('市场状态时间序列')
            ax.set_xlabel('日期')
            ax.set_ylabel('收盘价')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            return ax

    def _plot_transition_matrix_unified(self, ax=None, plotly=False):
        """统一的转移矩阵绘制"""
        state_order = self._get_unified_state_order()
        n = len(state_order)
        if self.model.transmat_.shape[0] != n:
            n = self.model.transmat_.shape[0]
            state_order = list(range(n))
        labels = [self.market_states.get(s, f"状态{s}").label if s in self.market_states else f"状态{s}" for s in state_order]
        transmat = self.model.transmat_
        if plotly:
            fig = go.Figure(go.Heatmap(
                z=transmat, 
                x=labels, 
                y=labels, 
                colorscale='Blues', 
                showscale=True,
                hovertemplate='从 %{y} 到 %{x}: %{z:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title="状态转移矩阵", 
                xaxis_title="转移到状态", 
                yaxis_title="来自状态"
            )
            return fig
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(transmat, annot=True, fmt='.3f', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('状态转移矩阵')
            ax.set_xlabel('转移到状态')
            ax.set_ylabel('来自状态')
            return ax

    def save_consistent_plots(self):
        """保存一致性的图表"""
        os.makedirs('output', exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_state_timeline_unified(ax=ax, plotly=False)
        plt.tight_layout()
        plt.savefig('output/state_timeline_unified.png', dpi=150, bbox_inches='tight')
        plt.close()
        fig_plotly = self._plot_state_timeline_unified(plotly=True)
        fig_plotly.write_html('output/state_timeline_unified.html')
        logger.info("统一格式的图表已保存")

    def _debug_transition_matrix_data(self):
        """调试转移矩阵数据"""
        logger.info("=== 转移矩阵调试信息 ===")
        logger.info(f"模型状态数: {self.model.n_components}")
        logger.info(f"转移矩阵形状: {self.model.transmat_.shape}")
        logger.info(f"market_states键: {list(self.market_states.keys())}")
        logger.info(f"实际观测到的状态: {np.unique(self.states)}")
        logger.info(f"转移矩阵内容:")
        logger.info(self.model.transmat_)
        logger.info(f"转移矩阵是否包含NaN: {np.isnan(self.model.transmat_).any()}")
        logger.info(f"转移矩阵是否包含Inf: {np.isinf(self.model.transmat_).any()}")
        logger.info("========================")

    def save_full_plotly_html_report_fixed(self, html_path='output/state_analysis_plotly_report_fixed.html'):
        """
        修复版本：生成每个分析图都为独立Plotly交互式图的HTML报告
        """
        import plotly.graph_objs as go
        import plotly.io as pio
        import os
        import numpy as np

        # 调试信息
        self._debug_transition_matrix_data()

        # 1. 市场状态时间序列
        df_plot = pd.DataFrame({
            'date': self.index,
            'close': self.df.loc[self.index, 'Close'].values,
            'state': [self.market_states.get(int(s), f"状态{int(s)}").label for s in self.states]
        })
        
        color_map = {
            '牛市': '#FFD700',
            '熊市': '#1E90FF',
            '震荡市': '#A9A9A9',
            '高波动': '#FF6347',
            '低波动': '#32CD32',
            '强牛市': '#FF8C00',
            '其它': '#CCCCCC'
        }
        
        traces_timeline = []
        for state in df_plot['state'].unique():
            sub = df_plot[df_plot['state'] == state]
            if len(sub) > 0:  # 确保有数据
                traces_timeline.append(go.Scatter(
                    x=sub['date'], y=sub['close'],
                    mode='markers',
                    name=state,
                    marker=dict(size=6, color=color_map.get(state, '#888')),
                    hovertemplate='日期: %{x}<br>收盘价: %{y}<br>状态: ' + state + '<extra></extra>'
                ))
        
        fig_timeline = go.Figure(traces_timeline)
        fig_timeline.update_layout(title="市场状态时间序列", xaxis_title="日期", yaxis_title="收盘价")

        # 2. 状态转移矩阵 - 修复版本
        transmat = self.model.transmat_.copy()
        n_components = self.model.n_components
        
        labels = []
        for i in range(n_components):
            if i in self.market_states:
                label = self.market_states[i].label
                labels.append(f"{label}({i})")
            else:
                labels.append(f"状态{i}")

        # 打印调试信息
        logger.info("=== DEBUG: 状态转移矩阵可视化 ===")
        logger.info("labels:", labels)
        logger.info("transmat.shape:", transmat.shape)
        logger.info("transmat (rounded):\n", np.round(transmat, 3))
        logger.info("row_sums:", transmat.sum(axis=1))
        logger.info("market_states:", {k: v.label for k, v in self.market_states.items()})
        logger.info("states in data:", np.unique(self.states))
        logger.info("==============================")

        fig_heatmap = go.Figure(go.Heatmap(
            z=transmat, 
            x=labels, 
            y=labels, 
            colorscale='Blues', 
            showscale=True,
            hovertemplate='从 %{y} 到 %{x}: %{z:.3f}<extra></extra>',
            text=np.round(transmat, 3),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig_heatmap.update_layout(
            title="状态转移矩阵", 
            xaxis_title="转移到状态", 
            yaxis_title="来自状态",
            width=600,
            height=500
        )

        # 3. 年化收益率与波动率 - 使用实际存在的状态
        existing_states = [s for s in range(n_components) if s in self.market_states]
        if not existing_states:
            existing_states = list(self.market_states.keys())
        
        labels_stats = [f"{self.market_states[s].label}({s})" for s in existing_states]
        returns = [self.market_states[s].mean_return * 252 for s in existing_states]
        vols = [self.market_states[s].volatility * np.sqrt(252) for s in existing_states]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=labels_stats, 
            y=returns, 
            name='年化收益率', 
            marker_color='skyblue',
            yaxis='y'
        ))
        fig_bar.add_trace(go.Scatter(
            x=labels_stats, 
            y=vols, 
            name='年化波动率', 
            yaxis='y2', 
            mode='lines+markers', 
            marker_color='red',
            line=dict(width=3)
        ))
        fig_bar.update_layout(
            title="年化收益率与波动率",
            xaxis_title="市场状态",
            yaxis=dict(title="年化收益率", side='left'),
            yaxis2=dict(title="年化波动率", overlaying='y', side='right'),
            legend=dict(x=0.7, y=1)
        )

        # 4. 风险收益散点图
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=vols, 
            y=returns, 
            mode='markers+text',
            text=labels_stats, 
            textposition='top center',
            marker=dict(size=16, color='gold', line=dict(width=2, color='black')),
            name='风险收益',
            hovertemplate='状态: %{text}<br>年化波动率: %{x:.2%}<br>年化收益率: %{y:.2%}<extra></extra>'
        ))
        fig_scatter.update_layout(
            title="风险收益散点图", 
            xaxis_title="年化波动率", 
            yaxis_title="年化收益率"
        )

        # 5. 平均持续天数
        durations = [self.market_states[s].avg_duration for s in existing_states]
        fig_duration = go.Figure()
        fig_duration.add_trace(go.Bar(
            x=labels_stats, 
            y=durations, 
            name='平均持续天数', 
            marker_color='lightgreen',
            hovertemplate='状态: %{x}<br>平均持续天数: %{y:.1f}<extra></extra>'
        ))
        fig_duration.update_layout(
            title="各状态平均持续天数", 
            xaxis_title="市场状态", 
            yaxis_title="平均持续天数"
        )

        # 6. 年化收益率分布
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns, 
            nbinsx=min(20, len(returns)), 
            name='年化收益率分布', 
            marker_color='skyblue',
            hovertemplate='年化收益率: %{x:.2%}<br>频数: %{y}<extra></extra>'
        ))
        fig_hist.update_layout(
            title="各状态年化收益率分布", 
            xaxis_title="年化收益率", 
            yaxis_title="频数"
        )

        # 7. 详细文本报告
        report_html = f"<pre style='background:#f8f8f8; padding:10px; font-family:monospace;'>{self.generate_report()}</pre>"

        # 组装HTML
        html_content = f"""
        <html>
        <head>
            <meta charset=\"utf-8\">
            <title>HMM市场状态交互式分析报告（修复版）</title>
            <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    margin-top: 40px;
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                .info-box {{
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>HMM 市场状态交互式分析报告（修复版）</h1>
            <div class="info-box">
                <strong>分析期间:</strong> {self.index.min().strftime('%Y-%m-%d')} 至 {self.index.max().strftime('%Y-%m-%d')}<br>
                <strong>总样本数:</strong> {len(self.states)}<br>
                <strong>模型状态数:</strong> {self.n_states}<br>
                <strong>实际状态数:</strong> {len(self.market_states)}
            </div>
            <hr>
            
            <h2>市场状态时间序列</h2>
            {pio.to_html(fig_timeline, full_html=False, include_plotlyjs=False)}
            
            <h2>状态转移矩阵</h2>
            {pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False)}
            
            <h2>年化收益率与波动率</h2>
            {pio.to_html(fig_bar, full_html=False, include_plotlyjs=False)}
            
            <h2>风险收益散点图</h2>
            {pio.to_html(fig_scatter, full_html=False, include_plotlyjs=False)}
            
            <h2>各状态平均持续天数</h2>
            {pio.to_html(fig_duration, full_html=False, include_plotlyjs=False)}
            
            <h2>各状态年化收益率分布</h2>
            {pio.to_html(fig_hist, full_html=False, include_plotlyjs=False)}
            
            <h2>详细文本报告</h2>
            {report_html}
            
        </body>
        </html>
        """

        # 保存文件
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"修复版Plotly交互式分析报告已保存到: {html_path}")
        return html_path


    def save_full_plotly_html_report_fixed_v2(self, html_path='output/state_analysis_plotly_report_fixed_v2.html'):
        """
        修复版本V2：解决转移矩阵在HTML中显示为空的问题
        """
        import plotly.graph_objs as go
        import plotly.io as pio
        import os
        import numpy as np

        # 调试信息
        self._debug_transition_matrix_data()

        # 1. 市场状态时间序列
        df_plot = pd.DataFrame({
            'date': self.index,
            'close': self.df.loc[self.index, 'Close'].values,
            'state': [self.market_states.get(int(s), f"状态{int(s)}").label for s in self.states]
        })
        
        color_map = {
            '牛市': '#FFD700',
            '熊市': '#1E90FF',
            '震荡市': '#A9A9A9',
            '高波动': '#FF6347',
            '低波动': '#32CD32',
            '强牛市': '#FF8C00',
            '其它': '#CCCCCC'
        }
        
        traces_timeline = []
        for state in df_plot['state'].unique():
            sub = df_plot[df_plot['state'] == state]
            if len(sub) > 0:
                traces_timeline.append(go.Scatter(
                    x=sub['date'], y=sub['close'],
                    mode='markers',
                    name=state,
                    marker=dict(size=6, color=color_map.get(state, '#888')),
                    hovertemplate='日期: %{x}<br>收盘价: %{y}<br>状态: ' + state + '<extra></extra>'
                ))
        
        fig_timeline = go.Figure(traces_timeline)
        fig_timeline.update_layout(
            title="市场状态时间序列", 
            xaxis_title="日期", 
            yaxis_title="收盘价",
            height=400
        )

        # 2. 状态转移矩阵 - 重点修复这部分
        transmat = self.model.transmat_.copy()
        n_components = self.model.n_components
        
        # 确保转移矩阵数据有效
        if np.any(np.isnan(transmat)) or np.any(np.isinf(transmat)):
            logger.info("警告：转移矩阵包含无效值，进行清理...")
            transmat = np.nan_to_num(transmat, nan=0.0, posinf=1.0, neginf=0.0)

        z = np.array(transmat, dtype=np.float64)
        logger.warning(f"z dtype: {z.dtype}, z shape: {z.shape}")

        # 生成标签
        labels = []
        for i in range(n_components):
            if i in self.market_states:
                label = self.market_states[i].label
                labels.append(f"{label}({i})")
            else:
                labels.append(f"状态{i}")

        logger.info("=== 转移矩阵修复调试 ===")
        logger.info("labels:", labels)
        logger.info("transmat.shape:", transmat.shape)
        logger.info("transmat min/max:", transmat.min(), transmat.max())
        logger.info("transmat是否包含极小值:", np.any(transmat < 1e-10))
        
        # 处理极小值问题 - 这可能是导致显示为空的原因
        transmat_display = transmat.copy()
        transmat_display[transmat_display < 1e-10] = 0  # 将极小值设为0
        
        # 创建自定义的文本标注
        text_annotations = []
        for i in range(transmat_display.shape[0]):
            text_row = []
            for j in range(transmat_display.shape[1]):
                val = transmat_display[i, j]
                if val < 0.001:
                    text_row.append("0.000")
                else:
                    text_row.append(f"{val:.3f}")
            text_annotations.append(text_row)
        
        # 方法1：使用go.Heatmap with explicit text
        fig_heatmap = go.Figure(go.Heatmap(
            z=transmat_display,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            zmin=0,
            zmax=1,
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "black"},
            hovertemplate='从 %{y} 到 %{x}: %{z:.3f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title="状态转移矩阵",
            xaxis_title="转移到状态",
            yaxis_title="来自状态",
            width=700,
            height=600,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0)
        )

        # 方法2：备用的Plotly Table作为转移矩阵显示
        table_data = []
        table_data.append(['状态'] + labels)  # 表头
        for i, from_state in enumerate(labels):
            row = [from_state] + [f"{transmat_display[i,j]:.3f}" for j in range(len(labels))]
            table_data.append(row)
        
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=table_data[0],
                    fill_color='paleturquoise',
                    align='center',
                    font_size=12),
            cells=dict(values=list(zip(*table_data[1:])),
                    fill_color='lavender',
                    align='center',
                    font_size=11))
        ])
        fig_table.update_layout(title="状态转移矩阵（表格形式）", height=400)

        # 3. 年化收益率与波动率
        existing_states = [s for s in range(n_components) if s in self.market_states]
        if not existing_states:
            existing_states = list(self.market_states.keys())
        
        labels_stats = [f"{self.market_states[s].label}({s})" for s in existing_states]
        returns = [self.market_states[s].mean_return * 252 for s in existing_states]
        vols = [self.market_states[s].volatility * np.sqrt(252) for s in existing_states]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=labels_stats,
            y=returns,
            name='年化收益率',
            marker_color='skyblue',
            yaxis='y'
        ))
        fig_bar.add_trace(go.Scatter(
            x=labels_stats,
            y=vols,
            name='年化波动率',
            yaxis='y2',
            mode='lines+markers',
            marker_color='red',
            line=dict(width=3)
        ))
        fig_bar.update_layout(
            title="年化收益率与波动率",
            xaxis_title="市场状态",
            yaxis=dict(title="年化收益率", side='left'),
            yaxis2=dict(title="年化波动率", overlaying='y', side='right'),
            legend=dict(x=0.7, y=1),
            height=400
        )

        # 4. 风险收益散点图
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=vols,
            y=returns,
            mode='markers+text',
            text=labels_stats,
            textposition='top center',
            marker=dict(size=16, color='gold', line=dict(width=2, color='black')),
            name='风险收益',
            hovertemplate='状态: %{text}<br>年化波动率: %{x:.2%}<br>年化收益率: %{y:.2%}<extra></extra>'
        ))
        fig_scatter.update_layout(
            title="风险收益散点图",
            xaxis_title="年化波动率",
            yaxis_title="年化收益率",
            height=400
        )

        # 5. 平均持续天数
        durations = [self.market_states[s].avg_duration for s in existing_states]
        fig_duration = go.Figure()
        fig_duration.add_trace(go.Bar(
            x=labels_stats,
            y=durations,
            name='平均持续天数',
            marker_color='lightgreen',
            hovertemplate='状态: %{x}<br>平均持续天数: %{y:.1f}<extra></extra>'
        ))
        fig_duration.update_layout(
            title="各状态平均持续天数",
            xaxis_title="市场状态",
            yaxis_title="平均持续天数",
            height=400
        )

        # 6. 年化收益率分布
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns,
            nbinsx=min(20, len(returns)),
            name='年化收益率分布',
            marker_color='skyblue',
            hovertemplate='年化收益率: %{x:.2%}<br>频数: %{y}<extra></extra>'
        ))
        fig_hist.update_layout(
            title="各状态年化收益率分布",
            xaxis_title="年化收益率",
            yaxis_title="频数",
            height=400
        )

        # 7. 详细文本报告
        report_html = f"<pre style='background:#f8f8f8; padding:10px; font-family:monospace; white-space: pre-wrap;'>{self.generate_report()}</pre>"

        # 转换图表为HTML字符串 - 使用更稳定的方法
        try:
            timeline_html = pio.to_html(fig_timeline, full_html=False, include_plotlyjs='cdn', div_id="timeline")
            heatmap_html = pio.to_html(fig_heatmap, full_html=False, include_plotlyjs='cdn', div_id="heatmap")
            table_html = pio.to_html(fig_table, full_html=False, include_plotlyjs='cdn', div_id="table")
            bar_html = pio.to_html(fig_bar, full_html=False, include_plotlyjs='cdn', div_id="bar")
            scatter_html = pio.to_html(fig_scatter, full_html=False, include_plotlyjs='cdn', div_id="scatter")
            duration_html = pio.to_html(fig_duration, full_html=False, include_plotlyjs='cdn', div_id="duration")
            hist_html = pio.to_html(fig_hist, full_html=False, include_plotlyjs='cdn', div_id="hist")
        except Exception as e:
            logger.info(f"图表转换HTML时出错: {e}")
            # 提供备用显示
            timeline_html = "<p>时间序列图生成失败</p>"
            heatmap_html = "<p>热力图生成失败</p>"
            table_html = "<p>表格生成失败</p>"
            bar_html = "<p>柱状图生成失败</p>"
            scatter_html = "<p>散点图生成失败</p>"
            duration_html = "<p>持续时间图生成失败</p>"
            hist_html = "<p>直方图生成失败</p>"

        # 组装HTML - 更加详细的调试信息
        debug_info = f"""
        <div class="debug-info" style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3>调试信息</h3>
            <p><strong>转移矩阵形状:</strong> {transmat.shape}</p>
            <p><strong>转移矩阵数值范围:</strong> [{transmat.min():.6f}, {transmat.max():.6f}]</p>
            <p><strong>状态标签:</strong> {', '.join(labels)}</p>
            <p><strong>极小值处理:</strong> 将小于1e-10的值设为0显示</p>
        </div>
        """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>HMM市场状态交互式分析报告（修复版V2）</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    text-align: center;
                }}
                h2 {{
                    margin-top: 40px;
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                .info-box {{
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .chart-container {{
                    margin: 20px 0;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>HMM 市场状态交互式分析报告（修复版V2）</h1>
                
                <div class="info-box">
                    <strong>分析期间:</strong> {self.index.min().strftime('%Y-%m-%d')} 至 {self.index.max().strftime('%Y-%m-%d')}<br>
                    <strong>总样本数:</strong> {len(self.states)}<br>
                    <strong>模型状态数:</strong> {self.n_states}<br>
                    <strong>实际状态数:</strong> {len(self.market_states)}
                </div>
                
                {debug_info}
                
                <hr>
                
                <h2>📈 市场状态时间序列</h2>
                <div class="chart-container">
                    {timeline_html}
                </div>
                
                <h2>🔄 状态转移矩阵（热力图）</h2>
                <div class="chart-container">
                    {heatmap_html}
                </div>
                
                <h2>📋 状态转移矩阵（表格形式）</h2>
                <div class="chart-container">
                    {table_html}
                </div>
                
                <h2>📊 年化收益率与波动率</h2>
                <div class="chart-container">
                    {bar_html}
                </div>
                
                <h2>💼 风险收益散点图</h2>
                <div class="chart-container">
                    {scatter_html}
                </div>
                
                <h2>⏰ 各状态平均持续天数</h2>
                <div class="chart-container">
                    {duration_html}
                </div>
                
                <h2>📈 各状态年化收益率分布</h2>
                <div class="chart-container">
                    {hist_html}
                </div>
                
                <h2>📄 详细文本报告</h2>
                {report_html}
                
            </div>
        </body>
        </html>
        """

        # 保存文件
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"修复版V2 Plotly交互式分析报告已保存到: {html_path}")
        logger.info("主要修复：")
        logger.info("1. 处理了转移矩阵中的极小值显示问题")
        logger.info("2. 添加了表格形式的转移矩阵作为备用显示")
        logger.info("3. 增加了详细的调试信息")
        logger.info("4. 改进了HTML结构和样式")
        
        return html_path

    # 将此方法添加到IntelligentStateAnalyzer类中
    # 使用方法：
    # analyzer.save_full_plotly_html_report_fixed_v2()