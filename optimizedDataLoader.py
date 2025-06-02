import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Union

warnings.filterwarnings('ignore')

from loguru import logger


class OptimizedDataLoader:
    """优化的数据加载器"""
    
    @staticmethod
    def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
        """加载并标准化市场数据"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件类型加载数据
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        logger.info(f"原始数据列名: {list(df.columns)}")
        
        # 自动检测并设置日期索引
        date_columns = ['Date', 'date', 'timestamp', 'time', 'DateTime', 'DATE', '日期', '时间']
        date_col = next((col for col in date_columns if col in df.columns), None)
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            logger.warning("未找到日期列，使用默认索引")
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # 标准化列名
        column_mapping = {
            'Open': ['Open', 'open', 'OPEN', '开盘价', 'open_price', 'Open_Price', 'price', 'adj_open', 'Adj Open', '开盘'],
            'Close': ['Close', 'close', 'CLOSE', '收盘价', 'close_price', 'Close_Price', 'price', 'adj_close', 'Adj Close', '收盘'],
            'High': ['High', 'high', 'HIGH', '最高价', 'high_price', 'High_Price', '最高'],
            'Low': ['Low', 'low', 'LOW', '最低价', 'low_price', 'Low_Price', '最低'],
            'Volume': ['Volume', 'volume', 'VOLUME', '成交量', 'volume_price', 'Vol', 'vol', '成交']
        }
        
        # 创建新的DataFrame来存储标准化后的数据
        standardized_df = pd.DataFrame(index=df.index)
        
        # 映射列名并复制数据
        for standard_name, possible_names in column_mapping.items():
            found = False
            for col in possible_names:
                if col in df.columns:
                    standardized_df[standard_name] = df[col]
                    found = True
                    logger.info(f"找到列 {standard_name} 的映射: {col} -> {standard_name}")
                    break
            if not found:
                logger.error(f"未找到列 {standard_name} 的任何可能名称")
                logger.error(f"可接受的列名: {possible_names}")
                raise ValueError(f"找不到必需的列: {standard_name}。可接受的列名: {possible_names}")
        
        # 数据清洗
        standardized_df = standardized_df.replace([np.inf, -np.inf], np.nan)
        standardized_df = standardized_df.dropna(subset=['Close'])
        
        logger.info(f"数据加载成功: {len(standardized_df)} 条记录")
        logger.info(f"标准化后的列名: {list(standardized_df.columns)}")
        
        return standardized_df.sort_index()

