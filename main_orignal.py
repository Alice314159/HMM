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
from EnhancedHMMOptimizer import EnhancedHMMOptimizer

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
    original_df = loader.load_data(pipeline.config['data']['path'])

    try:
               
        # 步骤0: 按时间分割数据
        logger.info("按时间分割数据...")
        train_df, test_df = pipeline.split_data_by_time(original_df)

        # 保存Close列
        train_close = train_df['Close'].copy()
        test_close = test_df['Close'].copy()

        # 步骤1: 计算特征
        logger_seperator("特征计算")
        featured_df = pipeline.compute_features(train_df)
        
        # 获取特征名称
        feature_names = featured_df.columns.tolist()
        
        # 步骤2: 训练scalar计算,并保存scaler，返回scalar计算后的数据和index
        X_train, train_index = pipeline.normalize_data(featured_df)
        
        # 步骤3: 拟合PCA
        X_train_processed = pipeline.apply_pca(X_train)
        
        # 存储处理后的特征名称
        pipeline.processed_feature_names = feature_names

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
        
        # 恢复Close列
        train_df['Close'] = train_close
        test_df['Close'] = test_close
        
        return pipeline, validation_result, train_df, test_df, X_train_processed, train_index
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise



def predict_with_model(config_path: str = "config.yaml", 
                      pipeline_path: str = "models/hmm_pipeline"):
    """
    使用训练好的模型进行预测
    
    参数:
    - config_path: 配置文件路径
    - pipeline_path: 训练好的流水线路径
    """
    try:
        # 初始化流水线
        pipeline = ImprovedHMMPipeline(config_path)
        
        # 加载训练好的流水线
        logger.info(f"加载训练好的流水线: {pipeline_path}")
        pipeline.load_pipeline(pipeline_path)
        
        # 加载原始数据
        logger.info("加载原始数据...")
        loader = OptimizedDataLoader()
        original_df = loader.load_data(pipeline.config['data']['path'])
        
        # 按时间分割数据
        logger.info("按时间分割数据...")
        _, test_df_raw = pipeline.split_data_by_time(original_df)
        
        # 保存原始数据的完整副本（包含所有列）
        test_df_original = test_df_raw.copy()
        
        # 计算特征
        featured_test_df = pipeline.compute_features(test_df_raw)
        
        # 从配置文件获取特征选择设置
        use_manual_features = pipeline.config['use_manual_features']
        
        if use_manual_features:
            # 使用手动配置的特征集
            logger.info("使用手动配置的特征集...")
            hmm_features = pipeline.config['manual_features']
            
            # 检查特征是否存在
            available_features = [f for f in hmm_features if f in featured_test_df.columns]
            missing_features = [f for f in hmm_features if f not in featured_test_df.columns]
            
            if missing_features:
                logger.warning(f"以下特征在数据中不存在: {missing_features}")
            
            logger.info(f"使用 {len(available_features)} 个手动配置的特征进行预测")
            logger.info(f"选中的特征: {available_features}")
            
            # 只使用选定的特征
            featured_test_df = featured_test_df[available_features]
            
        else:
            # 使用自动特征选择
            logger.info("使用自动特征选择...")
            # 获取特征名称
            feature_names = featured_test_df.columns.tolist()
            
            # 对测试数据进行预处理
            X_test, test_index = pipeline.normalize_data(featured_test_df)
            
            # 使用与训练时相同的特征选择
            logger.info("应用特征选择...")
            optimizer = EnhancedHMMOptimizer()
            feature_selection_results = optimizer.comprehensive_feature_selection(
                X_test,
                test_df_original['Close'].values,  # 使用测试集的收盘价作为目标变量
                feature_names,
                n_features=pipeline.config['feature_selection_config']['n_features']
            )
            
            # 获取选中的特征
            selected_features = feature_selection_results['selected_features']
            logger.info(f"自动选择的特征数量: {len(selected_features)}")
            logger.info(f"选中的特征: {selected_features}")
            
            # 使用选中的特征
            X_test_selected = X_test[:, [feature_names.index(f) for f in selected_features]]
            featured_test_df = featured_test_df[selected_features]
        
        # 对测试数据进行预处理
        X_test, test_index = pipeline.normalize_data(featured_test_df)
        
        logger.info(f"X_test: {X_test.shape}")
        
        # 应用PCA转换
        X_test_processed = pipeline.apply_pca(X_test)
        
        # 预测测试数据
        test_states = pipeline.predict_states(X_test_processed)
        
        # 将预测的状态添加到完整数据中
        if len(test_states) == len(test_index):
            # 创建状态Series，使用正确的索引
            states_series = pd.Series(test_states, index=test_index, name='HMM_State')
            # 将状态添加到完整数据中
            test_df_original = test_df_original.join(states_series, how='left')
        
        return test_states, test_index, test_df_original, X_test_processed
        
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
    

    train_analyzer.save_full_plotly_html_report_fixed_v2('output/improved_train_plotly_report.html')
    test_analyzer.save_full_plotly_html_report_fixed_v2('output/improved_test_plotly_report.html')  
    
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


def train_model_improved(config_path: str = "config.yaml"):
    """
    改进的HMM模型训练流程
    
    参数:
    - config_path: 配置文件路径
    """
    # 创建必要目录
    create_directories()
    
    # 初始化流水线
    pipeline = ImprovedHMMPipeline(config_path)
    
    logger_seperator("开始HMM模型训练流程")
    
    # 加载原始数据
    logger.info("加载原始数据...")
    loader = OptimizedDataLoader()
    original_df = loader.load_data(pipeline.config['data']['path'])
    
    try:
        # 步骤0: 按时间分割数据
        logger.info("按时间分割数据...")
        train_df_raw, test_df_raw = pipeline.split_data_by_time(original_df)
        
        # 保存原始数据的完整副本（包含所有列）
        train_df_original = train_df_raw.copy()
        test_df_original = test_df_raw.copy()
        
        logger.info(f"原始训练数据形状: {train_df_original.shape}")
        logger.info(f"原始训练数据列: {train_df_original.columns.tolist()}")
        
        # 步骤1: 计算特征
        logger_seperator("特征计算")
        featured_train_df = pipeline.compute_features(train_df_raw)
        
        # 从配置文件获取特征选择设置
        use_manual_features = pipeline.config['use_manual_features']
        
        if use_manual_features:
            # 使用手动配置的特征集
            logger.info("使用手动配置的特征集...")
            hmm_features = pipeline.config['manual_features']
            
            # 检查特征是否存在
            available_features = [f for f in hmm_features if f in featured_train_df.columns]
            missing_features = [f for f in hmm_features if f not in featured_train_df.columns]
            
            if missing_features:
                logger.warning(f"以下特征在数据中不存在: {missing_features}")
            
            logger.info(f"使用 {len(available_features)} 个手动配置的特征进行训练")
            logger.info(f"选中的特征: {available_features}")
            
            # 只使用选定的特征
            featured_train_df = featured_train_df[available_features]
            
        else:
            # 使用自动特征选择
            logger.info("使用自动特征选择...")
            # 获取特征名称
            feature_names = featured_train_df.columns.tolist()
            
            # 步骤2: 数据标准化
            X_train, train_index = pipeline.normalize_data(featured_train_df)
            
            # 步骤3: 特征选择
            logger_seperator("特征选择")
            optimizer = EnhancedHMMOptimizer()
            feature_selection_results = optimizer.comprehensive_feature_selection(
                X_train, 
                train_df_original['Close'].values,  # 使用收盘价作为目标变量
                feature_names,
                n_features=pipeline.config['feature_selection_config']['n_features']
            )
            
            # 获取选中的特征
            selected_features = feature_selection_results['selected_features']
            logger.info(f"自动选择的特征数量: {len(selected_features)}")
            logger.info(f"选中的特征: {selected_features}")
            
            # 使用选中的特征
            X_train_selected = X_train[:, [feature_names.index(f) for f in selected_features]]
            featured_train_df = featured_train_df[selected_features]
            available_features = selected_features
        
        logger.info(f"特征计算后数据形状: {featured_train_df.shape}")
        logger.info(f"特征列: {featured_train_df.columns.tolist()}")
        
        # 数据标准化
        X_train, train_index = pipeline.normalize_data(featured_train_df)
        np.savetxt('X_train.csv', X_train, delimiter=',')

        logger.info(f"X_train: {X_train.shape}")

        # PCA降维
        logger_seperator("PCA降维")
        X_train_processed = pipeline.apply_pca(X_train)
        
        # 存储处理后的特征名称
        pipeline.processed_feature_names = available_features
        
        # 训练HMM模型
        pipeline.train_hmm_model(X_train_processed)
        
        # 生成发射矩阵报告
        logger.info("\n生成发射矩阵分析报告...")
        optimizer = EnhancedHMMOptimizer()
        optimizer.generate_emission_matrix_report(
            pipeline.hmm_model,
            pipeline.processed_feature_names,
            pipeline.config['output']['emission_matrix_path']
        )
        
        # 创建完整的训练数据集（包含原始数据和状态）
        complete_train_df = train_df_original.copy()
        
        # 将预测的状态添加到完整数据中
        if hasattr(pipeline, 'train_states') and len(pipeline.train_states) == len(train_index):
            # 创建状态Series，使用正确的索引
            states_series = pd.Series(pipeline.train_states, index=train_index, name='HMM_State')
            # 将状态添加到完整数据中
            complete_train_df = complete_train_df.join(states_series, how='left')
        
        # 验证数据一致性
        logger.info("\n验证数据处理一致性...")
        validation_result = pipeline.validate_pipeline_consistency(
            X_train_processed, X_train_processed
        )
        
        # 打印验证报告
        print("\n" + "="*60)
        print("数据处理一致性验证报告")
        print("="*60)
        print(f"✓ 原始训练数据形状: {train_df_original.shape}")
        print(f"✓ 特征数据形状: {featured_train_df.shape}")
        print(f"✓ 处理后数据形状: {X_train_processed.shape}")
        print(f"✓ 完整训练数据形状: {complete_train_df.shape}")
        print(f"✓ 特征维度匹配: {validation_result['shape_match']}")
        print(f"✓ 特征名称一致: {validation_result['feature_names_match']}")
        print("="*60)
        
        # 保存流水线
        logger.info("\n保存训练好的流水线...")
        pipeline.save_pipeline(pipeline.config['output']['model_path'])
        
        # 保存数据快照用于分析
        complete_train_df.to_csv('output/complete_train_data.csv')
        featured_train_df.to_csv('output/featured_train_data.csv')
        
        logger_seperator("流程结束")
        
        return (pipeline, validation_result, 
                complete_train_df, test_df_original, 
                X_train_processed, train_index, featured_train_df)
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise


def analyze_data_flow(original_df, featured_df, processed_array):
    """分析数据流转过程"""
    
    print("\n" + "="*60)
    print("数据流转分析")
    print("="*60)
    
    print(f"1. 原始数据:")
    print(f"   - 形状: {original_df.shape}")
    print(f"   - 列名: {original_df.columns.tolist()}")
    print(f"   - 时间范围: {original_df.index.min()} 到 {original_df.index.max()}")
    
    print(f"\n2. 特征工程后:")
    print(f"   - 形状: {featured_df.shape}")
    print(f"   - 列名: {featured_df.columns.tolist()}")
    
    print(f"\n3. 标准化+PCA后:")
    print(f"   - 形状: {processed_array.shape}")
    print(f"   - 数据类型: {type(processed_array)}")
    
    # 检查数据完整性
    original_cols = set(original_df.columns)
    featured_cols = set(featured_df.columns)
    
    lost_cols = original_cols - featured_cols
    new_cols = featured_cols - original_cols
    
    if lost_cols:
        print(f"\n⚠️  丢失的原始列: {list(lost_cols)}")
    if new_cols:
        print(f"✓ 新增的特征列: {list(new_cols)}")
    
    print("="*60)


# 使用示例
if __name__ == "__main__":
    config_file = r"E:\Vesper\HMM\config.yaml"
    
    # 训练模型
    (pipeline, validation_result, complete_train_df, test_df_original, 
     X_train_processed, train_index, featured_train_df) = train_model_improved(config_file)
    
    # 分析数据流转
    analyze_data_flow(
        complete_train_df[['Open', 'High', 'Low', 'Close', 'Volume']], 
        featured_train_df, 
        X_train_processed
    )

    complete_train_df.to_csv('output/complete_train_data.csv')
    featured_train_df.to_csv('output/featured_train_data.csv')  

    # 保存模型
    pipeline.save_pipeline('models/hmm_pipeline')

    # 预测
    test_states, test_index, test_df, X_test_processed = predict_with_model(config_file)

    # 生成综合分析报告
    generate_comprehensive_analysis(
        pipeline, complete_train_df, test_df, train_index, test_index,
        X_train_processed, X_test_processed, test_states
    )


