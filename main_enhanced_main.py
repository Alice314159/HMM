import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np

import warnings
from loguru import logger
import os
warnings.filterwarnings('ignore')

from IntelligentStateAnalyzer import IntelligentStateAnalyzer
from optimizedDataLoader import OptimizedDataLoader
from ImprovedHmmPipeLine import ImprovedHMMPipeline

def create_directories():
    """创建必要的目录"""
    directories = ['output', 'models', 'figures']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def logger_seperator(section: str):
    """日志分隔符"""
    logger.info("=" * 50)
    logger.info(section)
    logger.info("=" * 50)


def train_model(config_path: str = "src/HMM/config.yaml"):
    """训练HMM模型"""
    # 创建必要目录
    create_directories()

    # 初始化流水线
    pipeline = ImprovedHMMPipeline(config_path)

    logger_seperator("开始HMM模型训练流程")

    # 加载原始数据
    logger.info("加载原始数据...")
    loader = OptimizedDataLoader()
    original_df = loader.load_data(pipeline.config['data_path'])

    try:
        # 步骤0: 按时间分割数据
        logger.info("按时间分割数据...")
        train_df, test_df = pipeline.split_data_by_time(original_df)

        # 保存原始价格数据用于后续分析
        price_columns = ['Close', 'Open', 'High', 'Low']
        train_prices = train_df[price_columns].copy()
        test_prices = test_df[price_columns].copy()

        # 计算收益率和技术指标
        logger.info("计算收益率和技术指标...")
        # 计算日收益率
        train_df['returns'] = train_df['Close'].pct_change()
        test_df['returns'] = test_df['Close'].pct_change()
        
        # 计算波动率（20日）
        train_df['volatility'] = train_df['returns'].rolling(window=20).std()
        test_df['volatility'] = test_df['returns'].rolling(window=20).std()
        
        # 计算相对强弱指标（RSI，14日）
        delta = train_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        train_df['RSI'] = 100 - (100 / (1 + rs))
        
        delta = test_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        test_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        train_df['MA20'] = train_df['Close'].rolling(window=20).mean()
        train_df['std20'] = train_df['Close'].rolling(window=20).std()
        train_df['upper_band'] = train_df['MA20'] + (train_df['std20'] * 2)
        train_df['lower_band'] = train_df['MA20'] - (train_df['std20'] * 2)
        
        test_df['MA20'] = test_df['Close'].rolling(window=20).mean()
        test_df['std20'] = test_df['Close'].rolling(window=20).std()
        test_df['upper_band'] = test_df['MA20'] + (test_df['std20'] * 2)
        test_df['lower_band'] = test_df['MA20'] - (test_df['std20'] * 2)

        # 定义用于PCA的特征列表
        pca_features = [
            'returns', 'volatility', 'RSI', 
            'MA20', 'std20', 'upper_band', 'lower_band'
        ]
        
        # 保存特征列表到pipeline
        pipeline.pca_features = pca_features

        # 步骤1: 计算特征
        logger_seperator("特征计算")
        featured_df = pipeline.compute_features(train_df)
        
        # 确保使用相同的特征
        featured_df = featured_df[pca_features]
        logger.info(f"训练数据特征维度: {featured_df.shape}")
        logger.info(f"使用的特征: {featured_df.columns.tolist()}")
        
        # 获取特征名称并保存
        feature_names = featured_df.columns.tolist()
        pipeline.processed_feature_names = feature_names
        
        # 步骤2: 训练scalar计算,并保存scaler，返回scalar计算后的数据和index
        X_train, train_index = pipeline.normalize_data(featured_df)

        # 步骤3: 拟合PCA
        X_train_processed = pipeline.apply_pca(X_train)
        logger.info(f"PCA转换后的维度: {X_train_processed.shape}")

        # 步骤4: 训练HMM
        pipeline.train_hmm_model(X_train_processed)

        # 验证数据一致性
        logger.info("\n验证数据处理一致性...")
        validation_result = pipeline.validate_pipeline_consistency(
            X_train_processed, X_train_processed  # 使用训练数据验证
        )

        print("\n" + "="*60)
        print("数据处理一致性验证报告")
        print("="*60)
        print(f"✓ 训练数据形状: {validation_result['train_shape']}")
        print(f"✓ 特征维度匹配: {validation_result['shape_match']}")
        print(f"✓ 特征名称一致: {validation_result['feature_names_match']}")
        if 'statistics_check' in validation_result:
            stats = validation_result['statistics_check']
            print(f"✓ 均值差异合理: {stats['reasonable_mean_diff']}")
            print(f"✓ 标准差比例合理: {stats['reasonable_std_ratio']}")
        print("="*60)

        # 保存流水线
        logger.info("\n保存训练好的流水线...")
        pipeline.save_pipeline('models/hmm_pipeline')
        logger_seperator("流程结束")
        
        # 恢复原始价格数据用于分析
        for col in price_columns:
            train_df[col] = train_prices[col]
            test_df[col] = test_prices[col]
        
        return pipeline, validation_result, train_df, test_df, X_train_processed, train_index
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise



def predict_with_model(config_path: str = "src/HMM/config.yaml", 
                      pipeline_path: str = "models/hmm_pipeline"):
    """使用训练好的模型进行预测"""
    try:
        # 初始化流水线
        pipeline = ImprovedHMMPipeline(config_path)
        
        # 加载训练好的流水线
        logger.info(f"加载训练好的流水线: {pipeline_path}")
        pipeline.load_pipeline(pipeline_path)
        
        # 加载原始数据
        logger.info("加载原始数据...")
        loader = OptimizedDataLoader()
        original_df = loader.load_data(pipeline.config['data_path'])
        
        # 保存原始价格数据用于后续分析
        price_columns = ['Close', 'Open', 'High', 'Low']
        test_prices = original_df[price_columns].copy()
        
        # 计算收益率和技术指标
        logger.info("计算收益率和技术指标...")
        # 计算日收益率
        original_df['returns'] = original_df['Close'].pct_change()
        
        # 计算波动率（20日）
        original_df['volatility'] = original_df['returns'].rolling(window=20).std()
        
        # 计算相对强弱指标（RSI，14日）
        delta = original_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        original_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        original_df['MA20'] = original_df['Close'].rolling(window=20).mean()
        original_df['std20'] = original_df['Close'].rolling(window=20).std()
        original_df['upper_band'] = original_df['MA20'] + (original_df['std20'] * 2)
        original_df['lower_band'] = original_df['MA20'] - (original_df['std20'] * 2)
        
        # 计算特征
        featured_df = pipeline.compute_features(original_df)
        
        # 使用与训练时相同的特征
        if hasattr(pipeline, 'pca_features'):
            featured_df = featured_df[pipeline.pca_features]
            logger.info(f"使用保存的PCA特征: {pipeline.pca_features}")
            logger.info(f"预测数据特征维度: {featured_df.shape}")
        else:
            logger.warning("未找到保存的PCA特征列表，使用所有可用特征")
        
        # 获取特征名称
        feature_names = featured_df.columns.tolist()
        pipeline.processed_feature_names = feature_names
        
        # 按时间分割数据
        _, test_df = pipeline.split_data_by_time(featured_df)
        
        # 对测试数据进行预处理
        X_test, test_index = pipeline.normalize_data(test_df)
        logger.info(f"标准化后的测试数据维度: {X_test.shape}")
        
        # 应用PCA转换
        X_test_processed = pipeline.apply_pca(X_test)
        logger.info(f"PCA转换后的测试数据维度: {X_test_processed.shape}")
        
        # 预测测试数据
        test_states = pipeline.predict_states(X_test_processed)
        
        # 恢复原始价格数据用于分析
        for col in price_columns:
            test_df[col] = test_prices[col]
        
        return test_states, test_index, test_df, X_test_processed
        
    except Exception as e:
        logger.error(f"预测过程失败: {e}")
        raise



def generate_comprehensive_analysis(pipeline, train_df, test_df, train_index, test_index,
                                  X_train_processed, X_test_processed, test_states):
    """生成综合分析报告"""
    
    # 训练集分析
    train_analyzer = IntelligentStateAnalyzer(
        pipeline.hmm_model, pipeline.train_states, X_train_processed,
        pipeline.processed_feature_names, train_index, train_df
    )
    
    # 测试集分析
    test_analyzer = IntelligentStateAnalyzer(
        pipeline.hmm_model, test_states, X_test_processed,
        pipeline.processed_feature_names, test_index, test_df
    )
    
    # 生成报告
    train_report = train_analyzer.generate_report()
    test_report = test_analyzer.generate_report()
    
    # 保存分析图表
    train_analyzer.save_all_figures(prefix='output/improved_train_')
    test_analyzer.save_all_figures(prefix='output/improved_test_')
    
    # 保存HTML报告
    train_analyzer.save_html_report('output/improved_train_report.html', 'output/improved_train_')
    test_analyzer.save_html_report('output/improved_test_report.html', 'output/improved_test_')
    
    logger.info("✓ 综合分析报告生成完成")


def load_and_predict_new_data(pipeline_path: str, new_data_path: str) -> np.ndarray:
    """使用已保存的流水线预测新数据"""
    
    # 加载流水线
    pipeline = ImprovedHMMPipeline("")
    pipeline.load_pipeline(pipeline_path)
    
    # 加载新数据
    loader = OptimizedDataLoader()
    new_df = loader.load_data(new_data_path)
    
    # 步骤1: 计算特征（使用已保存的特征工程器）
    featured_df = pipeline.step1_compute_features(new_df)
    
    # 步骤7-8: 预处理和预测
    X_processed, data_index = pipeline.step7_transform_test_data(featured_df)
    predictions = pipeline.step8_predict_test_data(X_processed)
    
    return predictions, data_index


if __name__ == "__main__":
    # Train the model
    pipeline, validation_result, train_df, test_df, X_train_processed, train_index = train_model()
    
    # Evaluate the model on test data
    logger_seperator("开始模型评估")
    test_states, test_index, test_df, X_test_processed = predict_with_model()
    
    # Generate comprehensive analysis
    logger_seperator("生成综合分析报告")
    generate_comprehensive_analysis(
        pipeline=pipeline,
        train_df=train_df,
        test_df=test_df,
        train_index=train_index,
        test_index=test_index,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        test_states=test_states
    )