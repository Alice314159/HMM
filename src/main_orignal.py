from ImprovedHmmPipeLine import ImprovedHMMPipelineCls
from src.analysis.IntelligentStateAnalyzer import IntelligentStateAnalyzer
from src.utils.configLoader import HMMConfigReader
def run_train(config: HMMConfigReader):
    """
    运行主程序
    :param config_path: 配置文件路径
    """
    # 创建并运行改进的HMM管道
    pipeline = ImprovedHMMPipelineCls(config)
    X_train, feature_names,train_df,index = pipeline.prepare_training_data()
    # Train HMM model on training data
    pipeline.train_hmm_model(X_train)
    # 创建并运行智能状态分析器
    analyzer = IntelligentStateAnalyzer(pipeline.hmm_model, pipeline.train_states,X_train, feature_names, index,train_df)
    analyzer.save_full_plotly_html_report(config.output.train_report_path)
    analyzer.generate_emission_matrix_report(config.output.emission_matrix_path)





def run_test(config:HMMConfigReader):
    """
    运行测试程序
    :param config_path: 配置文件路径
    """
    # 创建并运行改进的HMM管道
    pipeline = ImprovedHMMPipelineCls(config)

    # Prepare testing data
    X_test, feature_names,test_df,index = pipeline.prepare_testing_data()

    test_nstate = pipeline.predict_states(X_test)

    # 创建并运行智能状态分析器
    analyzer = IntelligentStateAnalyzer(pipeline.hmm_model, test_nstate, X_test, feature_names, index,
                                        test_df)
    analyzer.save_full_plotly_html_report(config.output.test_report_path)



if __name__ == "__main__":
    config_path = r"E:\Vesper\HMM0\src\config.yaml"
    config = HMMConfigReader(config_path)  # Load configuration

    run_train(config)
    run_test(config)
