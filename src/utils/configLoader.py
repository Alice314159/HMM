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
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


@dataclass
class FeatureSelectionConfig:
    """Feature selection configuration parameters"""
    enable: bool = True
    n_features: int = 10
    methods: List[str] = field(default_factory=lambda: ['f_test', 'mutual_info', 'random_forest', 'pca', 'correlation'])


@dataclass
# class TrainerConfig:
#     """HMM trainer configuration parameters"""
#     n_states: int = 3
#     n_components: int = 4
#     covariance_type: str = 'full'
#     n_iter: int = 100
#     tol: float = 0.01
#     verbose: int = 2
#     n_jobs: int = -1
#     max_iter: int = 300
#     k_range: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 35, 40])
#     state_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10])
class TrainerConfig:
    """Enhanced HMM trainer configuration parameters with optimization features"""

    # Original HMM parameters
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

    # New optimization parameters
    selection_strategy: str = 'composite'  # 'aic', 'bic', 'composite', 'cross_validation', 'log_likelihood'
    max_attempts: int = 20
    early_stopping_patience: int = 5
    min_improvement: float = 0.001
    use_cross_validation: bool = False
    cv_folds: int = 3

    # Composite scoring weights (for selection_strategy='composite')
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'log_likelihood': 0.3,
        'aic': -0.2,  # Negative because lower is better
        'bic': -0.2,  # Negative because lower is better
        'silhouette': 0.15,
        'calinski_harabasz': 0.1,
        'stability': 0.05
    })

    # Advanced training parameters
    use_progressive_training: bool = True
    use_data_stabilization: bool = True
    min_state_proportion: float = 0.01  # Minimum proportion of samples per state
    normalize_features: bool = False  # Whether to normalize features in later attempts

    # Validation parameters
    validate_model_quality: bool = True
    require_convergence: bool = True
    min_unique_states: int = 2

    # Parallel processing
    use_parallel_training: bool = False
    parallel_jobs: int = -1

    # Reproducibility
    random_seed: int = 42
    set_random_state: bool = True

    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate selection strategy
        valid_strategies = ['aic', 'bic', 'composite', 'cross_validation', 'log_likelihood']
        if self.selection_strategy not in valid_strategies:
            raise ValueError(f"selection_strategy must be one of {valid_strategies}")

        # Validate ranges
        if not self.state_range:
            raise ValueError("state_range cannot be empty")
        if min(self.state_range) < 2:
            raise ValueError("Minimum number of states must be at least 2")

        # Validate weights for composite strategy
        if self.selection_strategy == 'composite':
            if not self.score_weights:
                raise ValueError("score_weights cannot be empty for composite strategy")

            # Ensure all required weights are present
            required_weights = ['log_likelihood', 'aic', 'bic', 'silhouette', 'calinski_harabasz', 'stability']
            missing_weights = set(required_weights) - set(self.score_weights.keys())
            if missing_weights:
                raise ValueError(f"Missing score weights: {missing_weights}")

        # Validate cross-validation parameters
        if self.use_cross_validation and self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2 for cross-validation")

        # Validate numerical parameters
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be non-negative")
        if self.min_state_proportion <= 0 or self.min_state_proportion >= 1:
            raise ValueError("min_state_proportion must be between 0 and 1")

    def get_hmm_params(self) -> Dict:
        """Get standard HMM parameters for model initialization"""
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'verbose': self.verbose,
            'random_state': self.random_seed if self.set_random_state else None
        }

    def get_optimization_params(self) -> Dict:
        """Get optimization-specific parameters"""
        return {
            'selection_strategy': self.selection_strategy,
            'max_attempts': self.max_attempts,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement': self.min_improvement,
            'use_cross_validation': self.use_cross_validation,
            'cv_folds': self.cv_folds,
            'score_weights': self.score_weights.copy(),
            'use_progressive_training': self.use_progressive_training,
            'use_data_stabilization': self.use_data_stabilization,
            'min_state_proportion': self.min_state_proportion,
            'normalize_features': self.normalize_features,
            'validate_model_quality': self.validate_model_quality,
            'require_convergence': self.require_convergence,
            'min_unique_states': self.min_unique_states
        }

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'n_states': self.n_states,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
            'max_iter': self.max_iter,
            'k_range': self.k_range,
            'state_range': self.state_range,
            'selection_strategy': self.selection_strategy,
            'max_attempts': self.max_attempts,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement': self.min_improvement,
            'use_cross_validation': self.use_cross_validation,
            'cv_folds': self.cv_folds,
            'score_weights': self.score_weights,
            'use_progressive_training': self.use_progressive_training,
            'use_data_stabilization': self.use_data_stabilization,
            'min_state_proportion': self.min_state_proportion,
            'normalize_features': self.normalize_features,
            'validate_model_quality': self.validate_model_quality,
            'require_convergence': self.require_convergence,
            'min_unique_states': self.min_unique_states,
            'use_parallel_training': self.use_parallel_training,
            'parallel_jobs': self.parallel_jobs,
            'random_seed': self.random_seed,
            'set_random_state': self.set_random_state
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainerConfig':
        """Create configuration from dictionary"""
        # Handle potential missing keys by using defaults
        filtered_dict = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name in config_dict:
                filtered_dict[field_name] = config_dict[field_name]
            elif field_def.default is not dataclass.MISSING:
                filtered_dict[field_name] = field_def.default
            elif field_def.default_factory is not dataclass.MISSING:
                filtered_dict[field_name] = field_def.default_factory()

        return cls(**filtered_dict)

    def update_score_weights(self, new_weights: Dict[str, float]):
        """Update score weights for composite strategy"""
        self.score_weights.update(new_weights)
        self._validate_config()

    def get_parameter_grid(self) -> List[Dict]:
        """Get parameter grid for grid search"""
        base_params = [
            {'covariance_type': 'diag', 'n_iter': 100, 'tol': 1e-3, 'algorithm': 'viterbi'},
            {'covariance_type': 'diag', 'n_iter': 200, 'tol': 1e-4, 'algorithm': 'viterbi'},
            {'covariance_type': 'tied', 'n_iter': 150, 'tol': 1e-3, 'algorithm': 'viterbi'},
            {'covariance_type': 'spherical', 'n_iter': 100, 'tol': 1e-3, 'algorithm': 'viterbi'},
            {'covariance_type': 'full', 'n_iter': 200, 'tol': 1e-4, 'algorithm': 'viterbi'},
        ]

        # Add advanced parameters if progressive training is enabled
        if self.use_progressive_training:
            base_params.extend([
                {'covariance_type': 'diag', 'n_iter': 300, 'tol': 1e-5, 'algorithm': 'baum-welch'},
                {'covariance_type': 'tied', 'n_iter': 250, 'tol': 1e-4, 'algorithm': 'baum-welch'},
            ])

        return base_params


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