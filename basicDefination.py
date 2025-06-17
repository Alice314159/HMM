import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor
import joblib
import os
import plotly.graph_objs as go
import plotly.io as pio
import plotly.subplots as sp

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from hmmlearn.hmm import GaussianHMM
from scipy import stats

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """模型评估指标"""
    log_likelihood: float
    aic: float
    bic: float
    silhouette_score: float
    calinski_harabasz_score: float
    converged: bool
    n_params: int

# =============================================================================
# 列名定义
# =============================================================================

class ColumnNames:
    """标准列名定义"""
    
    # 价格相关列名
    CLOSE = 'Close'
    HIGH = 'High'
    LOW = 'Low'
    OPEN = 'Open'
    VOLUME = 'Volume'

    
    # 特征列名
    RETURNS = 'returns'
    LOG_RETURNS = 'log_returns'
    ABS_RETURNS = 'abs_returns'
    VOLATILITY = 'volatility'
    REALIZED_VOL = 'realized_vol'
    VOL_RATIO = 'vol_ratio'
    GARCH_VOL = 'garch_vol'
    
    # 技术指标列名
    MA_RATIO = 'ma_ratio'
    MA_DISTANCE = 'ma_distance'
    MOMENTUM = 'momentum'
    ROC = 'roc'
    RSI = 'rsi'
    MACD = 'macd'
    MACD_SIGNAL = 'macd_signal'
    MACD_HIST = 'macd_hist'
    KDJ_K = 'kdj_k'
    KDJ_D = 'kdj_d'
    KDJ_J = 'kdj_j'
    BB_WIDTH = 'bb_width'
    
    # 趋势指标列名
    ADX = 'adx'
    MA_SLOPE = 'ma_slope'
    
    # 价差/结构指标列名
    HL_SPREAD = 'hl_spread'
    OC_SPREAD = 'oc_spread'
    OHLC_CENTER = 'ohlc_center'
    CLOSE_ZSCORE = 'close_zscore'
    
    # 成交量指标列名
    VOLUME_STD = 'volume_std'
    VOLUME_ZSCORE = 'volume_zscore'
    
    # 结构稳定性指标列名
    MA_SLOPE_RATIO = 'ma_slope_ratio'
    PRICE_MA_CORR = 'price_ma_corr'
    
    # 波动率特征列名
    VOLATILITY_WINDOWS = [5, 10, 20, 60]
    REALIZED_VOL_WINDOWS = [5, 10, 20, 60]
    BB_WINDOWS = [20, 50]
    MA_WINDOWS = [5, 10, 20, 60]
    VOLUME_WINDOWS = [5, 10, 20]
    RETURNS_WINDOWS = [1,5, 10, 20, 60]
    
    @classmethod
    def get_volatility_column(cls, window: int) -> str:
        """获取波动率列名"""
        return f"{cls.VOLATILITY}_{window}"
    
    @classmethod
    def get_realized_vol_column(cls, window: int) -> str:
        """获取已实现波动率列名"""
        return f"{cls.REALIZED_VOL}_{window}"
    
    @classmethod
    def get_ma_column(cls, window: int, is_ratio: bool = True) -> str:
        """获取移动平均列名"""
        prefix = cls.MA_RATIO if is_ratio else cls.MA_DISTANCE
        return f"{prefix}_{window}"
    
    @classmethod
    def get_momentum_column(cls, period: int) -> str:
        """获取动量指标列名"""
        return f"{cls.MOMENTUM}_{period}"

    @classmethod
    def get_returns_column(cls, window: int, section: str = "log") -> str:
        """获取收益率列名"""
        if section == "abs":
            return f"{cls.ABS_RETURNS}_{window}"
        elif section == "log":
            return f"{cls.LOG_RETURNS}_{window}"
        else:
            return f"{cls.RETURNS}_{window}"
    
    @classmethod
    def get_roc_column(cls, period: int) -> str:
        """获取变化率列名"""
        return f"{cls.ROC}_{period}"
    
    @classmethod
    def get_rsi_column(cls, window: int) -> str:
        """获取RSI列名"""
        return f"{cls.RSI}_{window}"
        
    @classmethod
    def get_bb_width_column(cls, window: int) -> str:
        """获取布林带宽度列名"""
        return f"{cls.BB_WIDTH}_{window}"
        
    @classmethod
    def get_ma_slope_column(cls, window: int) -> str:
        """获取MA斜率列名"""
        return f"{cls.MA_SLOPE}_{window}"
        
    @classmethod
    def get_volume_std_column(cls, window: int) -> str:
        """获取成交量标准差列名"""
        return f"{cls.VOLUME_STD}_{window}"
        
    @classmethod
    def get_price_ma_corr_column(cls, window: int) -> str:
        """获取价格与MA相关性列名"""
        return f"{cls.PRICE_MA_CORR}_{window}"

# =============================================================================
# 市场状态定义
# =============================================================================

@dataclass
class MarketState:
    """市场状态定义"""
    state_id: int
    label: str
    description: str
    mean_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    probability: float
    avg_duration: float
    risk_level: str

class MarketStates:
    """市场状态类型定义"""
    STRONG_BULL = "强牛市"
    BULL = "牛市"
    SIDEWAYS = "震荡市"
    BEAR = "熊市"
    HIGH_VOLATILITY = "高波动"
    
    @classmethod
    def get_state_description(cls, state_type: str) -> str:
        """获取状态描述"""
        descriptions = {
            cls.STRONG_BULL: "高收益低波动的理想环境",
            cls.BULL: "正收益且风险调整后表现良好",
            cls.SIDEWAYS: "收益率接近0，波动适中",
            cls.BEAR: "负收益或大回撤",
            cls.HIGH_VOLATILITY: "收益方向不明且波动放大"
        }
        return descriptions.get(state_type, "未知状态")

# =============================================================================
# 配置相关定义
# =============================================================================

class ConfigDefaults:
    """配置默认值"""
    SCALER_TYPE = 'robust'
    FEATURE_SELECTION = False
    TOP_K_FEATURES = 10
    PCA_VARIANCE_RATIO = 0.95
    HMM_N_COMPONENTS = 3
    HMM_COVARIANCE_TYPE = 'full'
    HMM_RANDOM_STATE = 42
    HMM_N_ITER = 100
    HMM_TOL = 1e-3

# =============================================================================
# 常量定义
# =============================================================================

class Constants:
    """常量定义"""
    EPSILON = 1e-8  # 数值稳定性
    MAX_RETURN = 0.3  # 最大收益率限制
    MIN_RETURN = -0.3  # 最小收益率限制
    TRADING_DAYS_PER_YEAR = 252  # 年交易日数
    RISK_FREE_RATE = 0.03  # 无风险利率

