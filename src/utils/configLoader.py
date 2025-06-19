import yaml
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration parameters"""
    path: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class FeatureSelectionConfig:
    """Feature selection configuration parameters"""
    enable: bool = True
    n_features: int = 10
    methods: List[str] = field(default_factory=lambda: ['f_test', 'mutual_info', 'random_forest', 'pca', 'correlation'])


@dataclass
class TrainerConfig:
    """HMM trainer configuration parameters"""
    n_states: int = 3
    n_components: int = 4
    covariance_type: str = 'full'
    n_iter: int = 100
    tol: float = 0.01
    verbose: int = 2
    n_jobs: int = -1
    max_iter: int = 300
    k_range: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 35, 40])
    state_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10])


@dataclass
class OutputConfig:
    """Output paths configuration"""
    model_path: str = '../models/hmm_pipeline.pkl'
    scalar_path: str = '../models/scaler.pkl'
    feature_selection_path: str = '../models/feature_selection.pkl'
    pca_path: str = '../models/pca_model.pkl'
    feature_analysis_path: str = '../output/feature_analysis_report.txt'
    emission_matrix_path: str = '../output/emission_matrix_report.html'
    train_report_path: str = '../output/train_report.html'
    test_report_path: str = '../output/test_report.html'
    aic_bic_path: str = '../output/aic_bic_history.png'


class HMMConfigReader:
    """
    Configuration reader for HMM market state identification system.

    This class reads and validates YAML configuration files containing
    parameters for data processing, feature engineering, model training,
    and output generation.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration reader.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = None

        # Main configuration parameters
        self.scaler: str = 'robust'
        self.feature_selection: bool = False
        self.top_k_features: int = 30
        self.pca_variance_ratio: Union[int, float] = 0.95
        self.pca_components: Optional[Union[int, float]] = None
        self.use_manual_features: bool = True
        self.manual_features: List[str] = []

        # Configuration objects
        self.data: Optional[DataConfig] = None
        self.feature_selection_config: Optional[FeatureSelectionConfig] = None
        self.trainer: Optional[TrainerConfig] = None
        self.output: Optional[OutputConfig] = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        self.config_path = config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

        self._parse_config()
        self._validate_config()

    def _parse_config(self) -> None:
        """Parse the loaded configuration dictionary into class attributes."""
        if not self.config:
            raise ValueError("No configuration loaded")

        # Parse main parameters
        self.scaler_type = self.config.get('scaler', 'robust')
        self.feature_selection = self.config.get('feature_selection', False)
        self.top_k_features = self.config.get('top_k_features', 30)
        self.pca_variance_ratio = self.config.get('pca_variance_ratio', 0.95)
        self.pca_components = self.config.get('pca_components', None)
        self.use_manual_features = self.config.get('use_manual_features', True)
        self.manual_features = self.config.get('manual_features', [])

        # Parse data configuration
        data_config = self.config.get('data', {})
        self.data = DataConfig(**data_config)

        # Parse feature selection configuration
        fs_config = self.config.get('feature_selection_config', {})
        self.feature_selection_config = FeatureSelectionConfig(**fs_config)

        # Parse trainer configuration
        trainer_config = self.config.get('trainer', {})
        self.trainer = TrainerConfig(**trainer_config)

        # Parse output configuration
        output_config = self.config.get('output', {})
        self.output = OutputConfig(**output_config)


    def _validate_config(self) -> None:
        """Validate the parsed configuration parameters."""
        # Validate scaler type
        valid_scalers = ['standard', 'minmax', 'robust']
        if self.scaler not in valid_scalers:
            raise ValueError(f"Invalid scaler type: {self.scaler}. Must be one of {valid_scalers}")

        # Validate PCA parameters
        if isinstance(self.pca_variance_ratio, float) and not (0 < self.pca_variance_ratio <= 1):
            raise ValueError("pca_variance_ratio must be between 0 and 1 when specified as float")

        # Validate data file exists
        if not os.path.exists(self.data.path):
            raise FileNotFoundError(f"Data file not found: {self.data.path}")

        # Validate date formats (basic check)
        date_fields = [self.data.train_start, self.data.train_end,
                       self.data.test_start, self.data.test_end]
        for date_str in date_fields:
            if not self._is_valid_date_format(date_str):
                raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

        # Validate trainer parameters
        if self.trainer.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.trainer.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.trainer.tol <= 0:
            raise ValueError("tol must be positive")

    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if date string is in YYYY-MM-DD format."""
        try:
            from datetime import datetime
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def create_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        output_paths = [
            self.output.model_path,
            self.output.scalar_path,
            self.output.feature_selection_path,
            self.output.pca_path,
            self.output.report_path,
            self.output.feature_analysis_path,
            self.output.emission_matrix_path,
            self.output.train_report_path,
            self.output.test_report_path
        ]

        for path in output_paths:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")

    def get_feature_list(self) -> List[str]:
        """Get the list of features to use based on configuration."""
        if self.use_manual_features:
            return self.manual_features
        else:
            # Return empty list if not using manual features
            # The actual feature selection will be handled by the feature selection module
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'scaler': self.scaler,
            'feature_selection': self.feature_selection,
            'top_k_features': self.top_k_features,
            'pca_variance_ratio': self.pca_variance_ratio,
            'pca_components': self.pca_components,
            'use_manual_features': self.use_manual_features,
            'manual_features': self.manual_features,
            'data': {
                'path': self.data.path,
                'train_start': self.data.train_start,
                'train_end': self.data.train_end,
                'test_start': self.data.test_start,
                'test_end': self.data.test_end
            },
            'feature_selection_config': {
                'enable': self.feature_selection_config.enable,
                'n_features': self.feature_selection_config.n_features,
                'methods': self.feature_selection_config.methods
            },
            'trainer': {
                'n_states': self.trainer.n_states,
                'n_components': self.trainer.n_components,
                'covariance_type': self.trainer.covariance_type,
                'n_iter': self.trainer.n_iter,
                'tol': float(self.trainer.get('tol', 1e-2)),
                'verbose': self.trainer.verbose,
                'n_jobs': self.trainer.n_jobs,
                'max_iter': self.trainer.max_iter,
                'k_range': self.trainer.k_range,
                'state_range': self.trainer.state_range
            },
            'output': {
                'model_path': self.output.model_path,
                'scalar_path': self.output.scalar_path,
                'feature_selection_path': self.output.feature_selection_path,
                'pca_path': self.output.pca_path,
                'report_path': self.output.report_path,
                'feature_analysis_path': self.output.feature_analysis_path,
                'emission_matrix_path': self.output.emission_matrix_path,
                'train_report_path': self.output.train_report_path,
                'test_report_path': self.output.test_report_path
            }
        }

    def __str__(self) -> str:
        """String representation of the configuration."""
        if not self.config:
            return "HMMConfigReader: No configuration loaded"

        return f"""HMMConfigReader Configuration:
        - Scaler: {self.scaler}
        - Feature Selection: {self.feature_selection}
        - Top K Features: {self.top_k_features}
        - PCA Variance Ratio: {self.pca_variance_ratio}
        - Use Manual Features: {self.use_manual_features}
        - Manual Features: {len(self.manual_features)} features
        - Data Path: {self.data.path}
        - Train Period: {self.data.train_start} to {self.data.train_end}
        - Test Period: {self.data.test_start} to {self.data.test_end}
        - HMM States: {self.trainer.n_states}
        - HMM Components: {self.trainer.n_components}"""

    def pca_config(self) -> Dict[str, Any]:
        """Get PCA configuration parameters."""
        return {
            'variance_ratio': self.pca_variance_ratio,
            'n_components': self.pca_components
        }

    def get_raw_data_path(self) -> str:
        """Get the raw data file path."""
        return self.data.path

    def get_trainer_config(self) -> Dict[str, Any]:
        """Get the trainer configuration parameters."""
        return {
            'n_states': self.trainer.n_states,
            'n_components': self.trainer.n_components,
            'covariance_type': self.trainer.covariance_type,
            'n_iter': self.trainer.n_iter,
            'tol': self.trainer.tol,
            'verbose': self.trainer.verbose,
            'n_jobs': self.trainer.n_jobs,
            'max_iter': self.trainer.max_iter
        }

# Example usage
if __name__ == "__main__":
    # Example of how to use the configuration reader
    config_reader = HMMConfigReader()

    # Load configuration from file
    try:
        config_reader.load_config("hmm_config.yaml")
        print("Configuration loaded successfully!")
        print(config_reader)

        # Create output directories
        config_reader.create_output_directories()

        # Access specific configuration values
        print(f"\nData file: {config_reader.data.path}")
        print(f"Manual features: {config_reader.get_feature_list()}")
        print(f"HMM states: {config_reader.trainer.n_states}")

    except Exception as e:
        print(f"Error loading configuration: {e}")