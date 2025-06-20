import pandas as pd
import numpy as np
import yaml
from src.FeatureManager.enhanced_feature_engineer import EnhancedFeatureEngineer
from src.utils.optimizedDataLoader import OptimizedDataLoader
from src.hmm.RobustHMMTrainer import RobustHMMTrainer
from loguru import logger
import joblib
from src.utils.configLoader import HMMConfigReader
class ImprovedHMMPipelineCls:
    """Improved HMM Training Pipeline following standard ML workflow"""

    def __init__(self,config:HMMConfigReader ):
        # Load configuration file
        self.config = config

        # Initialize feature processor
        self.feature_engineer = EnhancedFeatureEngineer(self.config)

        # Initialize PCA
        self.pca_variance_ratio = self.config.pca_variance_ratio
        self.pca_n_components = self.config.pca_components

        # Initialize other components
        self.hmm_trainer = None
        self.hmm_model = None

        # Record processed feature information
        self.feature_names = None
        self.train_index = None
        self.test_index = None

        # Training state records
        self.train_states = None
        self.train_metrics = None

        # Pipeline state
        self.is_fitted = False

    def prepare_training_data(self, cutoff_date: str = '2025-01-01') -> tuple:
        """Prepare training and testing data"""
        logger.info("Preparing training and testing data...")

        # Load original data
        original_df = self.load_data()

        # Split data by time
        train_df, _,_ = self.split_data_by_time(original_df, cutoff_date)

        # Compute features for training and testing data
        X_train, feature_names,index = self.feature_engineer.fit_transform(train_df)

        return X_train, feature_names,train_df,index


    def prepare_testing_data(self, cutoff_date: str = '2025-01-01') -> tuple:
        """Prepare testing data"""
        logger.info("Preparing testing data...")

        # Load original data
        original_df = self.load_data()

        # Split data by time
        _,_, test_df = self.split_data_by_time(original_df, cutoff_date)

        self.feature_engineer.load_pipeline(self.config.output.model_path)

        # Compute features for testing data
        X_test, feature_names, index = self.feature_engineer.transform(test_df)

        # Validate consistency between training and testing data
        validation_result = self.validate_pipeline_consistency(X_train=X_test, X_test=X_test)

        if not validation_result['shape_match']:
            raise ValueError("Training and testing data shapes do not match. Please check the preprocessing steps.")

        return X_test, feature_names,test_df, index

    def load_data(self) -> pd.DataFrame:
        loader = OptimizedDataLoader()
        original_df = loader.load_data(self.config.get_raw_data_path())
        return original_df

    def split_data_by_time(self, df: pd.DataFrame, cutoff_date: str = '2025-01-01') -> tuple:
        """Split data by time into training and testing sets"""
        logger.info("Splitting data by time...")

        train_start = pd.Timestamp(self.config.data.train_start)
        train_end = pd.Timestamp(self.config.data.train_end)

        valid_start = pd.Timestamp(self.config.data.valid_start)-pd.Timedelta(days=45)
        valid_end = pd.Timestamp(self.config.data.valid_end)

        test_start = pd.Timestamp(self.config.data.test_start)-pd.Timedelta(days=45)
        test_end = pd.Timestamp(self.config.data.test_end)



        train_df = df[(df.index <= train_end) & (df.index >= train_start)].copy()
        test_df = df[(df.index <= test_end) & (df.index >= test_start)].copy()
        valid_df = df[(df.index <= valid_end) & (df.index >= valid_start)].copy()


        logger.info(f"Training data: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
        logger.info(f"Testing data: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")

        return train_df,valid_df, test_df


    def train_hmm_model(self, X_train: np.ndarray) -> None:
        """Train HMM model on normalized training data"""
        logger.info("Training HMM model...")

        self.hmm_trainer = RobustHMMTrainer(self.config)

        # Train HMM model
        self.hmm_model, self.train_states, self.train_metrics = self.hmm_trainer.train(X_train)


        # Mark pipeline as fitted
        self.is_fitted = True

        logger.info("HMM model training completed")
        logger.info(f"Model state count: {self.hmm_model.n_components}")
        logger.info(f"Training data state distribution: {np.bincount(self.train_states)}")
        logger.info(f"Training metrics: {self.train_metrics}")

        self.hmm_trainer.save_model(self.hmm_model, "hmm_model.pkl")

        return self.hmm_model, self.train_states, self.train_metrics

    def predict_states(self, X_test: np.ndarray) -> np.ndarray:
        """Predict states for test data"""
        logger.info("Predicting states...")

        if self.hmm_model is None:
            self.hmm_model = joblib.load("hmm_model.pkl")
            #raise ValueError("HMM model not trained yet. Call train_hmm_model first.")

        # Predict test data states
        test_states = self.hmm_model.predict(X_test)

        logger.info("State prediction completed")
        logger.info(f"Test data state distribution: {np.bincount(test_states)}")

        return test_states



    def validate_pipeline_consistency(self, X_train: np.ndarray, X_test: np.ndarray) -> dict:
        """Validate consistency between training and test data processing"""
        logger.info("Validating data processing consistency...")

        validation_result = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'shape_match': X_train.shape[1] == X_test.shape[1],
            'feature_count_match': len(self.feature_names) if self.feature_names else 0,
            'feature_names_match': True  # Default to True
        }

        # Check feature names consistency
        if self.feature_names:
            validation_result['feature_names_match'] = (
                len(self.feature_names) == X_train.shape[1] and
                len(self.feature_names) == X_test.shape[1]
            )

        # Statistical characteristics check
        if validation_result['shape_match']:
            train_mean = np.mean(X_train, axis=0)
            test_mean = np.mean(X_test, axis=0)
            train_std = np.std(X_train, axis=0)
            test_std = np.std(X_test, axis=0)

            mean_diff = np.abs(train_mean - test_mean)
            std_ratio = test_std / (train_std + 1e-8)

            validation_result['statistics_check'] = {
                'mean_difference': mean_diff,
                'std_ratio': std_ratio,
                'reasonable_mean_diff': np.all(mean_diff < 2.0),
                'reasonable_std_ratio': np.all((std_ratio > 0.5) & (std_ratio < 2.0))
            }

            logger.info(f"Shape validation: {'✓' if validation_result['shape_match'] else '✗'}")
            logger.info(f"Feature names validation: {'✓' if validation_result['feature_names_match'] else '✗'}")
            if 'statistics_check' in validation_result:
                stats_check = validation_result['statistics_check']
                logger.info(f"Mean difference validation: {'✓' if stats_check['reasonable_mean_diff'] else '✗'}")
                logger.info(f"Std ratio validation: {'✓' if stats_check['reasonable_std_ratio'] else '✗'}")

        return validation_result

    def get_feature_importance(self) -> dict:
        """Get feature importance information"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")

        importance_info = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'n_states': self.hmm_model.n_components if self.hmm_model else 0,
            'train_state_distribution': np.bincount(self.train_states) if self.train_states is not None else None
        }
        importance_info = self.feature_processor.cal_pac(importance_info)
        return importance_info

