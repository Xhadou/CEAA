# CAAA: Context-Aware Anomaly Attribution
## Complete Implementation Plan for Coding Agent

---

# PROJECT OVERVIEW

**Goal**: Build a system that classifies microservice anomalies as either:
- `FAULT` (actual system issue requiring action)
- `EXPECTED_LOAD` (legitimate traffic spike, no action needed)  
- `UNKNOWN` (requires investigation)

**Novel Contribution**: First system to distinguish workload-induced anomalies from actual faults using external context.

**Timeline**: 4 weeks (can be compressed to 2 weeks for MVP)

---

# PHASE 0: ENVIRONMENT SETUP

## Step 0.1: Create Project Structure

```bash
mkdir -p ceaa-project/{data,src,models,notebooks,configs,outputs,tests}
cd ceaa-project

# Create subdirectories
mkdir -p data/{raw,processed,synthetic}
mkdir -p src/{data_loader,features,models,evaluation,utils}
mkdir -p models/{checkpoints,final}
mkdir -p outputs/{figures,results}
```

## Step 0.2: Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Deep Learning
torch>=2.0.0
pytorch-lightning>=2.0.0

# Time Series
tslearn>=0.6.0
stumpy>=1.12.0

# Anomaly Detection
pyod>=1.1.0

# Graph Neural Networks (optional for Week 3)
torch-geometric>=2.4.0
networkx>=3.1

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
python-dotenv>=1.0.0
requests>=2.31.0

# Evaluation
imbalanced-learn>=0.11.0

# Optional: LLM Integration
openai>=1.0.0
anthropic>=0.18.0
EOF

pip install -r requirements.txt
```

## Step 0.3: Create Configuration File

```bash
cat > configs/config.yaml << 'EOF'
# CEAA Configuration

data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  synthetic_path: "data/synthetic"
  
  # RCAEval dataset settings
  rcaeval:
    dataset: "RE1"  # RE1, RE2, or RE3
    systems: ["online-boutique", "sock-shop", "train-ticket"]
    
features:
  window_size: 60  # seconds
  stride: 10  # seconds
  
  workload_features:
    - global_load_ratio
    - cpu_request_correlation
    - cross_service_sync
    - error_rate_delta
    
  behavioral_features:
    - onset_gradient
    - cascade_pattern
    - recovery_indicator
    - affected_service_ratio

model:
  anomaly_detector:
    type: "lstm_ae"  # lstm_ae, vae, isolation_forest
    hidden_dim: 64
    latent_dim: 16
    num_layers: 2
    
  classifier:
    type: "random_forest"  # random_forest, gradient_boosting, mlp
    n_estimators: 100
    max_depth: 10
    
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
  
evaluation:
  test_split: 0.2
  val_split: 0.1
  metrics: ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
  
output:
  model_path: "models/final"
  results_path: "outputs/results"
  figures_path: "outputs/figures"
EOF
```

---

# PHASE 1: DATA ACQUISITION & PREPROCESSING

## Step 1.1: Download RCAEval Dataset

Create file: `src/data_loader/download_data.py`

```python
"""
Download and extract RCAEval dataset
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

RCAEVAL_URLS = {
    "RE1": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE1-online-boutique.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE1-sock-shop.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE1-train-ticket.zip"
    },
    "RE2": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE2-online-boutique.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE2-sock-shop.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE2-train-ticket.zip"
    }
}

def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # Clean up

def download_rcaeval_dataset(
    dataset: str = "RE1",
    systems: list = None,
    data_dir: str = "data/raw"
) -> None:
    """
    Download RCAEval dataset.
    
    Args:
        dataset: "RE1", "RE2", or "RE3"
        systems: List of systems to download. Default: all
        data_dir: Directory to save data
    """
    if systems is None:
        systems = ["online-boutique", "sock-shop", "train-ticket"]
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    urls = RCAEVAL_URLS.get(dataset, {})
    
    for system in systems:
        if system not in urls:
            print(f"Warning: {system} not found in {dataset}")
            continue
            
        url = urls[system]
        zip_path = data_path / f"{dataset}-{system}.zip"
        
        print(f"Downloading {dataset}/{system}...")
        download_file(url, zip_path)
        
        print(f"Extracting {dataset}/{system}...")
        extract_zip(zip_path, data_path / dataset / system)
        
    print("Download complete!")

if __name__ == "__main__":
    download_rcaeval_dataset(dataset="RE1", systems=["online-boutique"])
```

## Step 1.2: Data Loader for RCAEval Format

Create file: `src/data_loader/rcaeval_loader.py`

```python
"""
Load and parse RCAEval dataset format
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FailureCase:
    """Represents a single failure case from RCAEval."""
    case_id: str
    system: str
    service: str
    fault_type: str
    instance: int
    metrics: pd.DataFrame
    logs: Optional[pd.DataFrame] = None
    traces: Optional[pd.DataFrame] = None
    root_cause_service: Optional[str] = None
    root_cause_metric: Optional[str] = None
    label: str = "FAULT"  # Default label for RCAEval data

class RCAEvalLoader:
    """Load RCAEval datasets."""
    
    FAULT_TYPES = ["cpu", "mem", "disk", "delay", "loss", "socket"]
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def parse_case_name(self, case_name: str) -> Dict:
        """
        Parse case directory name.
        Format: {system}_{service}_{fault}_{instance}
        Example: online-boutique_frontend_cpu_1
        """
        parts = case_name.split("_")
        return {
            "system": parts[0],
            "service": parts[1],
            "fault_type": parts[2],
            "instance": int(parts[3]) if len(parts) > 3 else 0
        }
    
    def load_metrics(self, case_path: Path) -> pd.DataFrame:
        """Load metrics CSV file."""
        metrics_file = case_path / "metrics.csv"
        if not metrics_file.exists():
            # Try alternative naming
            for f in case_path.glob("*.csv"):
                if "metric" in f.name.lower():
                    metrics_file = f
                    break
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns and df.columns[0].lower() in ['time', 'ts']:
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            return df
        return pd.DataFrame()
    
    def load_logs(self, case_path: Path) -> Optional[pd.DataFrame]:
        """Load logs CSV file if available."""
        logs_file = case_path / "logs.csv"
        if logs_file.exists():
            return pd.read_csv(logs_file)
        return None
    
    def load_traces(self, case_path: Path) -> Optional[pd.DataFrame]:
        """Load traces CSV file if available."""
        traces_file = case_path / "traces.csv"
        if traces_file.exists():
            return pd.read_csv(traces_file)
        return None
    
    def load_ground_truth(self, case_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Load ground truth annotation if available."""
        gt_file = case_path / "ground_truth.json"
        if gt_file.exists():
            import json
            with open(gt_file) as f:
                gt = json.load(f)
            return gt.get("root_cause_service"), gt.get("root_cause_metric")
        return None, None
    
    def load_dataset(
        self,
        dataset: str = "RE1",
        system: str = "online-boutique",
        fault_types: Optional[List[str]] = None
    ) -> List[FailureCase]:
        """
        Load all failure cases from a dataset.
        
        Args:
            dataset: RE1, RE2, or RE3
            system: online-boutique, sock-shop, or train-ticket
            fault_types: Filter by fault types. None = all
            
        Returns:
            List of FailureCase objects
        """
        dataset_path = self.data_dir / dataset / system
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        cases = []
        
        for case_dir in sorted(dataset_path.iterdir()):
            if not case_dir.is_dir():
                continue
                
            info = self.parse_case_name(case_dir.name)
            
            # Filter by fault type
            if fault_types and info["fault_type"] not in fault_types:
                continue
            
            # Load data
            metrics = self.load_metrics(case_dir)
            logs = self.load_logs(case_dir)
            traces = self.load_traces(case_dir)
            rc_service, rc_metric = self.load_ground_truth(case_dir)
            
            case = FailureCase(
                case_id=case_dir.name,
                system=info["system"],
                service=info["service"],
                fault_type=info["fault_type"],
                instance=info["instance"],
                metrics=metrics,
                logs=logs,
                traces=traces,
                root_cause_service=rc_service,
                root_cause_metric=rc_metric,
                label="FAULT"  # All RCAEval cases are faults
            )
            cases.append(case)
        
        print(f"Loaded {len(cases)} failure cases from {dataset}/{system}")
        return cases

# Convenience function
def load_rcaeval(dataset="RE1", system="online-boutique") -> List[FailureCase]:
    """Quick loader function."""
    loader = RCAEvalLoader()
    return loader.load_dataset(dataset, system)

if __name__ == "__main__":
    # Test loading
    cases = load_rcaeval("RE1", "online-boutique")
    print(f"Loaded {len(cases)} cases")
    if cases:
        print(f"First case: {cases[0].case_id}")
        print(f"Metrics shape: {cases[0].metrics.shape}")
```

## Step 1.3: Generate Synthetic Load Spike Data

Create file: `src/data_loader/synthetic_load_generator.py`

```python
"""
Generate synthetic load spike scenarios to create EXPECTED_LOAD class.
This is KEY for the novel contribution - no existing dataset has this!
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import copy

@dataclass
class SyntheticLoadCase:
    """Represents a synthetic load spike case."""
    case_id: str
    system: str
    base_case_id: str  # Original case this was derived from
    load_multiplier: float
    event_type: str  # flash_sale, marketing_campaign, viral_content, scheduled_batch
    event_start: int  # Timestamp index
    event_duration: int  # Duration in samples
    metrics: pd.DataFrame
    context: dict
    label: str = "EXPECTED_LOAD"

class SyntheticLoadGenerator:
    """
    Generate synthetic load spike scenarios from normal operation data.
    
    The key insight: During legitimate load spikes:
    1. ALL services show proportional increase in resource usage
    2. CPU correlates strongly with request rate
    3. Error rates stay LOW (system is handling load correctly)
    4. Latency increases gradually, not suddenly
    """
    
    EVENT_TYPES = {
        "flash_sale": {
            "multiplier_range": (2.0, 5.0),
            "duration_range": (30, 120),  # samples
            "ramp_up_ratio": 0.1,  # 10% of duration for ramp-up
            "error_rate_increase": 0.01,  # Minimal error increase
        },
        "marketing_campaign": {
            "multiplier_range": (1.5, 3.0),
            "duration_range": (60, 240),
            "ramp_up_ratio": 0.2,
            "error_rate_increase": 0.005,
        },
        "scheduled_batch": {
            "multiplier_range": (1.3, 2.0),
            "duration_range": (20, 60),
            "ramp_up_ratio": 0.05,
            "error_rate_increase": 0.0,
        },
        "viral_content": {
            "multiplier_range": (3.0, 10.0),
            "duration_range": (60, 180),
            "ramp_up_ratio": 0.3,
            "error_rate_increase": 0.02,
        }
    }
    
    # Metrics that should scale with load
    LOAD_SENSITIVE_METRICS = [
        "cpu_usage", "cpu_utilization", "cpu",
        "memory_usage", "memory_utilization", "mem",
        "request_count", "requests", "throughput",
        "latency", "response_time", "duration",
        "network_in", "network_out", "bytes_sent", "bytes_received"
    ]
    
    # Metrics that should NOT scale with legitimate load
    FAULT_INDICATOR_METRICS = [
        "error_rate", "error_count", "errors",
        "5xx_count", "4xx_count",
        "timeout_count", "timeouts",
        "connection_refused", "connection_errors"
    ]
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def _identify_metric_columns(
        self, 
        df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """Identify which columns are load-sensitive, fault-indicator, or other."""
        load_cols = []
        fault_cols = []
        other_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(metric in col_lower for metric in self.LOAD_SENSITIVE_METRICS):
                load_cols.append(col)
            elif any(metric in col_lower for metric in self.FAULT_INDICATOR_METRICS):
                fault_cols.append(col)
            elif col_lower not in ['timestamp', 'time', 'ts']:
                other_cols.append(col)
                
        return load_cols, fault_cols, other_cols
    
    def _apply_load_spike(
        self,
        metrics: pd.DataFrame,
        multiplier: float,
        start_idx: int,
        duration: int,
        ramp_ratio: float = 0.1
    ) -> pd.DataFrame:
        """
        Apply load spike pattern to metrics.
        
        Creates realistic load pattern with:
        - Gradual ramp-up
        - Sustained high load
        - Gradual ramp-down
        """
        df = metrics.copy()
        load_cols, fault_cols, _ = self._identify_metric_columns(df)
        
        end_idx = min(start_idx + duration, len(df))
        ramp_samples = max(1, int(duration * ramp_ratio))
        
        for idx in range(start_idx, end_idx):
            # Calculate position in the load spike
            pos = idx - start_idx
            
            # Ramp-up phase
            if pos < ramp_samples:
                current_mult = 1.0 + (multiplier - 1.0) * (pos / ramp_samples)
            # Sustained phase
            elif pos < duration - ramp_samples:
                current_mult = multiplier + np.random.normal(0, 0.05)  # Small jitter
            # Ramp-down phase
            else:
                ramp_pos = pos - (duration - ramp_samples)
                current_mult = multiplier - (multiplier - 1.0) * (ramp_pos / ramp_samples)
            
            current_mult = max(1.0, current_mult)  # Ensure >= 1.0
            
            # Apply multiplier to load-sensitive metrics
            for col in load_cols:
                if pd.notna(df.loc[idx, col]):
                    # Add some noise for realism
                    noise = np.random.normal(0, 0.02 * current_mult)
                    df.loc[idx, col] = df.loc[idx, col] * current_mult * (1 + noise)
        
        # Ensure error metrics stay LOW (key differentiator from faults!)
        for col in fault_cols:
            if col in df.columns:
                # Keep error rates minimal during load spike
                df.loc[start_idx:end_idx, col] = df.loc[start_idx:end_idx, col] * 0.5
        
        return df
    
    def generate_from_normal(
        self,
        normal_metrics: pd.DataFrame,
        case_id: str,
        system: str,
        event_type: str = None,
        n_samples: int = 1
    ) -> List[SyntheticLoadCase]:
        """
        Generate synthetic load spike cases from normal operation data.
        
        Args:
            normal_metrics: DataFrame of normal operation metrics
            case_id: ID for the base case
            system: System name
            event_type: Type of load event (None = random)
            n_samples: Number of synthetic cases to generate
            
        Returns:
            List of SyntheticLoadCase objects
        """
        cases = []
        
        for i in range(n_samples):
            # Select event type
            if event_type is None:
                evt_type = np.random.choice(list(self.EVENT_TYPES.keys()))
            else:
                evt_type = event_type
            
            config = self.EVENT_TYPES[evt_type]
            
            # Generate parameters
            multiplier = np.random.uniform(*config["multiplier_range"])
            duration = np.random.randint(*config["duration_range"])
            
            # Choose start point (avoid beginning and end)
            max_start = len(normal_metrics) - duration - 10
            if max_start <= 10:
                continue
            start_idx = np.random.randint(10, max_start)
            
            # Apply load spike
            modified_metrics = self._apply_load_spike(
                normal_metrics,
                multiplier,
                start_idx,
                duration,
                config["ramp_up_ratio"]
            )
            
            # Create context information
            context = {
                "event_type": evt_type,
                "event_name": f"{evt_type.replace('_', ' ').title()} #{i+1}",
                "load_multiplier": multiplier,
                "start_index": start_idx,
                "duration": duration,
                "expected_cpu_increase": f"+{(multiplier-1)*100:.0f}%",
                "expected_latency_increase": "gradual",
                "expected_error_rate": "unchanged"
            }
            
            case = SyntheticLoadCase(
                case_id=f"synthetic_{case_id}_{evt_type}_{i}",
                system=system,
                base_case_id=case_id,
                load_multiplier=multiplier,
                event_type=evt_type,
                event_start=start_idx,
                event_duration=duration,
                metrics=modified_metrics,
                context=context,
                label="EXPECTED_LOAD"
            )
            cases.append(case)
        
        return cases
    
    def generate_from_fault_case(
        self,
        fault_metrics: pd.DataFrame,
        case_id: str,
        system: str
    ) -> SyntheticLoadCase:
        """
        Transform a fault case's pre-fault period into a load spike.
        
        This creates realistic scenarios where load happens before any fault.
        """
        # Use first half of metrics (before fault manifests)
        half_len = len(fault_metrics) // 2
        normal_portion = fault_metrics.iloc[:half_len].copy().reset_index(drop=True)
        
        # Generate load spike in this "normal" period
        cases = self.generate_from_normal(
            normal_portion,
            case_id,
            system,
            n_samples=1
        )
        
        return cases[0] if cases else None

def generate_balanced_dataset(
    fault_cases: List,  # FailureCase objects
    n_load_per_fault: int = 1,
    seed: int = 42
) -> Tuple[List, List]:
    """
    Generate a balanced dataset with both faults and load spikes.
    
    Args:
        fault_cases: List of FailureCase objects (from RCAEval)
        n_load_per_fault: Number of load spikes to generate per fault case
        
    Returns:
        Tuple of (all_cases, labels)
    """
    generator = SyntheticLoadGenerator(seed=seed)
    
    all_cases = []
    labels = []
    
    for fc in fault_cases:
        # Add original fault case
        all_cases.append(fc)
        labels.append("FAULT")
        
        # Generate corresponding load spike cases
        load_cases = generator.generate_from_normal(
            fc.metrics,
            fc.case_id,
            fc.system,
            n_samples=n_load_per_fault
        )
        
        for lc in load_cases:
            all_cases.append(lc)
            labels.append("EXPECTED_LOAD")
    
    print(f"Generated dataset: {len([l for l in labels if l == 'FAULT'])} faults, "
          f"{len([l for l in labels if l == 'EXPECTED_LOAD'])} load spikes")
    
    return all_cases, labels

if __name__ == "__main__":
    # Test generation
    from rcaeval_loader import load_rcaeval
    
    fault_cases = load_rcaeval("RE1", "online-boutique")[:5]  # First 5
    all_cases, labels = generate_balanced_dataset(fault_cases, n_load_per_fault=2)
    
    print(f"Total cases: {len(all_cases)}")
    print(f"Labels distribution: {pd.Series(labels).value_counts().to_dict()}")
```

---

# PHASE 2: FEATURE ENGINEERING

## Step 2.1: Workload Correlation Features

Create file: `src/features/workload_features.py`

```python
"""
Extract workload correlation features.
These features distinguish legitimate load from faults.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class WorkloadFeatures:
    """Container for workload-related features."""
    global_load_ratio: float
    cpu_request_correlation: float
    cross_service_sync: float
    error_rate_delta: float
    latency_cpu_correlation: float
    memory_trend_uniformity: float
    
    def to_dict(self) -> Dict:
        return {
            "global_load_ratio": self.global_load_ratio,
            "cpu_request_correlation": self.cpu_request_correlation,
            "cross_service_sync": self.cross_service_sync,
            "error_rate_delta": self.error_rate_delta,
            "latency_cpu_correlation": self.latency_cpu_correlation,
            "memory_trend_uniformity": self.memory_trend_uniformity
        }

class WorkloadFeatureExtractor:
    """
    Extract features that indicate workload patterns.
    
    Key insight: During legitimate load:
    - All services scale together (high cross_service_sync)
    - CPU correlates with requests (high cpu_request_correlation)
    - Error rates stay stable (low error_rate_delta)
    
    During faults:
    - Only affected services show anomalies (low cross_service_sync)
    - CPU may spike without corresponding requests
    - Error rates increase significantly
    """
    
    def __init__(self):
        # Patterns for identifying metric types
        self.cpu_patterns = ['cpu', 'processor']
        self.memory_patterns = ['mem', 'memory']
        self.request_patterns = ['request', 'req', 'throughput', 'qps', 'rps']
        self.latency_patterns = ['latency', 'response_time', 'duration', 'delay']
        self.error_patterns = ['error', 'err', '5xx', '4xx', 'fail']
    
    def _find_columns(self, df: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find columns matching any of the patterns."""
        matches = []
        for col in df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns):
                matches.append(col)
        return matches
    
    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute correlation with error handling."""
        if len(x) < 3 or len(y) < 3:
            return 0.0
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        
        corr, _ = stats.pearsonr(x, y)
        return corr if not np.isnan(corr) else 0.0
    
    def extract_global_load_ratio(
        self, 
        metrics: pd.DataFrame,
        window: Optional[slice] = None
    ) -> float:
        """
        Compute ratio of services showing increased load.
        
        High ratio (>0.8) suggests legitimate load affecting all services.
        Low ratio (<0.3) suggests localized issue (fault).
        """
        cpu_cols = self._find_columns(metrics, self.cpu_patterns)
        
        if not cpu_cols:
            return 0.5  # Default neutral
        
        df = metrics if window is None else metrics.iloc[window]
        
        # Compare second half to first half
        mid = len(df) // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]
        
        increased_count = 0
        for col in cpu_cols:
            if first_half[col].mean() > 0:
                ratio = second_half[col].mean() / first_half[col].mean()
                if ratio > 1.1:  # 10% increase threshold
                    increased_count += 1
        
        return increased_count / len(cpu_cols) if cpu_cols else 0.5
    
    def extract_cpu_request_correlation(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Compute correlation between CPU and request rate.
        
        High correlation (>0.7) suggests load-driven CPU usage.
        Low correlation (<0.3) suggests CPU issue not related to load.
        """
        cpu_cols = self._find_columns(metrics, self.cpu_patterns)
        req_cols = self._find_columns(metrics, self.request_patterns)
        
        if not cpu_cols or not req_cols:
            return 0.5
        
        # Use first CPU and request column found
        cpu_series = metrics[cpu_cols[0]].values
        req_series = metrics[req_cols[0]].values
        
        return abs(self._safe_correlation(cpu_series, req_series))
    
    def extract_cross_service_sync(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Measure how synchronized metric changes are across services.
        
        High sync (>0.7) suggests external factor (load) affecting all.
        Low sync (<0.3) suggests internal fault in specific service.
        """
        cpu_cols = self._find_columns(metrics, self.cpu_patterns)
        
        if len(cpu_cols) < 2:
            return 0.5
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(cpu_cols)):
            for j in range(i + 1, len(cpu_cols)):
                corr = self._safe_correlation(
                    metrics[cpu_cols[i]].values,
                    metrics[cpu_cols[j]].values
                )
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.5
    
    def extract_error_rate_delta(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Compute change in error rate.
        
        Low delta (<0.05) suggests system handling load well (legitimate load).
        High delta (>0.1) suggests system failing (fault).
        """
        error_cols = self._find_columns(metrics, self.error_patterns)
        
        if not error_cols:
            return 0.0
        
        mid = len(metrics) // 2
        first_half = metrics.iloc[:mid]
        second_half = metrics.iloc[mid:]
        
        deltas = []
        for col in error_cols:
            first_mean = first_half[col].mean()
            second_mean = second_half[col].mean()
            
            # Compute relative change
            if first_mean > 0:
                delta = (second_mean - first_mean) / first_mean
            else:
                delta = second_mean
            
            deltas.append(delta)
        
        return np.mean(deltas) if deltas else 0.0
    
    def extract_latency_cpu_correlation(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Correlation between latency and CPU.
        
        During legitimate load: high correlation (latency increases with CPU)
        During fault: may be low (latency spikes without CPU cause)
        """
        cpu_cols = self._find_columns(metrics, self.cpu_patterns)
        latency_cols = self._find_columns(metrics, self.latency_patterns)
        
        if not cpu_cols or not latency_cols:
            return 0.5
        
        return abs(self._safe_correlation(
            metrics[cpu_cols[0]].values,
            metrics[latency_cols[0]].values
        ))
    
    def extract_memory_trend_uniformity(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Check if memory trends are uniform across services.
        
        Uniform increase suggests load; localized spike suggests memory leak.
        """
        mem_cols = self._find_columns(metrics, self.memory_patterns)
        
        if len(mem_cols) < 2:
            return 0.5
        
        # Compute trend (slope) for each memory column
        trends = []
        x = np.arange(len(metrics))
        
        for col in mem_cols:
            y = metrics[col].values
            if np.std(y) > 0:
                slope, _, _, _, _ = stats.linregress(x, y)
                trends.append(slope)
        
        if not trends:
            return 0.5
        
        # Uniformity = 1 - coefficient of variation of slopes
        if np.mean(trends) != 0:
            cv = np.std(trends) / abs(np.mean(trends))
            return max(0, 1 - cv)
        
        return 0.5
    
    def extract_all(self, metrics: pd.DataFrame) -> WorkloadFeatures:
        """Extract all workload features."""
        return WorkloadFeatures(
            global_load_ratio=self.extract_global_load_ratio(metrics),
            cpu_request_correlation=self.extract_cpu_request_correlation(metrics),
            cross_service_sync=self.extract_cross_service_sync(metrics),
            error_rate_delta=self.extract_error_rate_delta(metrics),
            latency_cpu_correlation=self.extract_latency_cpu_correlation(metrics),
            memory_trend_uniformity=self.extract_memory_trend_uniformity(metrics)
        )

if __name__ == "__main__":
    # Test feature extraction
    from src.data_loader.rcaeval_loader import load_rcaeval
    
    cases = load_rcaeval("RE1", "online-boutique")[:3]
    extractor = WorkloadFeatureExtractor()
    
    for case in cases:
        features = extractor.extract_all(case.metrics)
        print(f"Case: {case.case_id}")
        print(f"  Features: {features.to_dict()}")
```

## Step 2.2: Behavioral Signature Features

Create file: `src/features/behavioral_features.py`

```python
"""
Extract behavioral signature features.
These capture HOW the anomaly manifests (sudden vs gradual, cascading vs localized).
"""
import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BehavioralFeatures:
    """Container for behavioral features."""
    onset_gradient: float  # Sudden (fault) vs gradual (load)
    peak_duration: float  # How long anomaly persists
    cascade_score: float  # Does anomaly spread across services?
    recovery_indicator: float  # Does system auto-recover?
    affected_service_ratio: float  # % of services affected
    variance_change_ratio: float  # Change in metric variance
    
    def to_dict(self) -> Dict:
        return {
            "onset_gradient": self.onset_gradient,
            "peak_duration": self.peak_duration,
            "cascade_score": self.cascade_score,
            "recovery_indicator": self.recovery_indicator,
            "affected_service_ratio": self.affected_service_ratio,
            "variance_change_ratio": self.variance_change_ratio
        }

class BehavioralFeatureExtractor:
    """
    Extract features that capture anomaly behavior patterns.
    
    Key insight:
    - Faults: sudden onset, localized, no auto-recovery
    - Load spikes: gradual onset, affects all services, may auto-recover
    """
    
    def __init__(self, anomaly_threshold: float = 2.0):
        """
        Args:
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.anomaly_threshold = anomaly_threshold
    
    def _detect_anomaly_indices(
        self, 
        series: np.ndarray,
        threshold: float = None
    ) -> np.ndarray:
        """Detect indices where values are anomalous (z-score > threshold)."""
        threshold = threshold or self.anomaly_threshold
        
        if np.std(series) == 0:
            return np.array([])
        
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return np.where(z_scores > threshold)[0]
    
    def extract_onset_gradient(
        self, 
        metrics: pd.DataFrame,
        target_cols: List[str] = None
    ) -> float:
        """
        Measure how suddenly the anomaly onset occurs.
        
        High gradient (>0.7) suggests sudden fault.
        Low gradient (<0.3) suggests gradual load increase.
        """
        if target_cols is None:
            # Use all numeric columns
            target_cols = metrics.select_dtypes(include=[np.number]).columns.tolist()
        
        gradients = []
        
        for col in target_cols:
            if col.lower() in ['timestamp', 'time', 'ts']:
                continue
                
            series = metrics[col].values
            anomaly_idx = self._detect_anomaly_indices(series)
            
            if len(anomaly_idx) == 0:
                continue
            
            # Look at the onset point
            first_anomaly = anomaly_idx[0]
            if first_anomaly < 3:
                continue
            
            # Compute gradient leading to first anomaly
            window = series[max(0, first_anomaly-5):first_anomaly+1]
            if len(window) > 1:
                gradient = np.abs(np.gradient(window)).max()
                # Normalize by series std
                if np.std(series) > 0:
                    gradient = gradient / np.std(series)
                gradients.append(gradient)
        
        return np.mean(gradients) if gradients else 0.5
    
    def extract_peak_duration(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Measure how long anomalous values persist.
        
        Normalized by total length. Long duration may indicate sustained load.
        """
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns
        
        durations = []
        
        for col in numeric_cols:
            if col.lower() in ['timestamp', 'time', 'ts']:
                continue
                
            series = metrics[col].values
            anomaly_idx = self._detect_anomaly_indices(series)
            
            if len(anomaly_idx) > 0:
                # Duration = span of anomaly indices
                duration = (anomaly_idx.max() - anomaly_idx.min() + 1) / len(series)
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def extract_cascade_score(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Measure if anomaly cascades (spreads with delay) across services.
        
        High cascade score suggests fault propagation.
        Simultaneous changes suggest external load.
        """
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns.tolist()
        
        # Find first anomaly index for each column
        first_anomalies = {}
        
        for col in numeric_cols:
            if col.lower() in ['timestamp', 'time', 'ts']:
                continue
                
            series = metrics[col].values
            anomaly_idx = self._detect_anomaly_indices(series)
            
            if len(anomaly_idx) > 0:
                first_anomalies[col] = anomaly_idx[0]
        
        if len(first_anomalies) < 2:
            return 0.0
        
        # Compute variance in onset times (high variance = cascade)
        onset_times = list(first_anomalies.values())
        
        # Normalize by total length
        variance = np.std(onset_times) / len(metrics)
        
        # Scale to 0-1 range
        return min(1.0, variance * 10)
    
    def extract_recovery_indicator(
        self, 
        metrics: pd.DataFrame,
        recovery_window: int = 10
    ) -> float:
        """
        Check if system shows signs of auto-recovery.
        
        High indicator (>0.7) suggests load spike that naturally subsides.
        Low indicator (<0.3) suggests persistent fault.
        """
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns
        
        recovery_scores = []
        
        for col in numeric_cols:
            if col.lower() in ['timestamp', 'time', 'ts']:
                continue
                
            series = metrics[col].values
            
            # Compare last window to peak
            if len(series) > recovery_window:
                peak_value = np.max(series)
                end_value = np.mean(series[-recovery_window:])
                baseline = np.mean(series[:recovery_window])
                
                if peak_value > baseline:
                    # Recovery = how much it dropped from peak towards baseline
                    recovery = (peak_value - end_value) / (peak_value - baseline + 1e-10)
                    recovery_scores.append(max(0, min(1, recovery)))
        
        return np.mean(recovery_scores) if recovery_scores else 0.5
    
    def extract_affected_service_ratio(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Compute ratio of columns showing anomalous behavior.
        
        High ratio (>0.7) suggests global impact (load).
        Low ratio (<0.3) suggests localized issue (fault).
        """
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns
        
        affected = 0
        total = 0
        
        for col in numeric_cols:
            if col.lower() in ['timestamp', 'time', 'ts']:
                continue
            
            total += 1
            series = metrics[col].values
            anomaly_idx = self._detect_anomaly_indices(series)
            
            if len(anomaly_idx) > 0:
                affected += 1
        
        return affected / total if total > 0 else 0.0
    
    def extract_variance_change_ratio(
        self, 
        metrics: pd.DataFrame
    ) -> float:
        """
        Measure change in variance between first and second half.
        
        High ratio suggests instability (fault).
        Moderate ratio suggests load (controlled increase).
        """
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns
        
        mid = len(metrics) // 2
        first_half = metrics.iloc[:mid]
        second_half = metrics.iloc[mid:]
        
        ratios = []
        
        for col in numeric_cols:
            if col.lower() in ['timestamp', 'time', 'ts']:
                continue
            
            var1 = first_half[col].var()
            var2 = second_half[col].var()
            
            if var1 > 0:
                ratio = var2 / var1
                ratios.append(ratio)
        
        if not ratios:
            return 1.0
        
        # Log transform to handle extreme values
        log_ratio = np.log1p(np.mean(ratios))
        
        # Normalize to 0-1 range (approximately)
        return min(1.0, log_ratio / 3)
    
    def extract_all(self, metrics: pd.DataFrame) -> BehavioralFeatures:
        """Extract all behavioral features."""
        return BehavioralFeatures(
            onset_gradient=self.extract_onset_gradient(metrics),
            peak_duration=self.extract_peak_duration(metrics),
            cascade_score=self.extract_cascade_score(metrics),
            recovery_indicator=self.extract_recovery_indicator(metrics),
            affected_service_ratio=self.extract_affected_service_ratio(metrics),
            variance_change_ratio=self.extract_variance_change_ratio(metrics)
        )

if __name__ == "__main__":
    # Test
    from src.data_loader.rcaeval_loader import load_rcaeval
    
    cases = load_rcaeval("RE1", "online-boutique")[:3]
    extractor = BehavioralFeatureExtractor()
    
    for case in cases:
        features = extractor.extract_all(case.metrics)
        print(f"Case: {case.case_id}")
        print(f"  Features: {features.to_dict()}")
```

## Step 2.3: Context Features

Create file: `src/features/context_features.py`

```python
"""
Extract context features from external event information.
This is the NOVEL component - integrating business context.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ContextFeatures:
    """Container for context-related features."""
    event_active: float  # Is a known event active?
    event_expected_impact: float  # Expected load multiplier
    time_seasonality: float  # Is this a typical high-traffic time?
    recent_deployment: float  # Was there a recent deployment?
    context_confidence: float  # Confidence in context information
    
    def to_dict(self) -> Dict:
        return {
            "event_active": self.event_active,
            "event_expected_impact": self.event_expected_impact,
            "time_seasonality": self.time_seasonality,
            "recent_deployment": self.recent_deployment,
            "context_confidence": self.context_confidence
        }

class EventCalendar:
    """
    Manages business events that could cause expected load spikes.
    
    In a real system, this would integrate with:
    - Marketing calendar APIs
    - Deployment pipelines (CI/CD)
    - Incident management systems
    """
    
    def __init__(self):
        self.events = []
        
    def add_event(
        self,
        name: str,
        event_type: str,
        start_time: datetime,
        end_time: datetime,
        expected_impact: float,
        affected_services: List[str] = None
    ):
        """Add an event to the calendar."""
        self.events.append({
            "name": name,
            "type": event_type,
            "start": start_time,
            "end": end_time,
            "impact": expected_impact,
            "services": affected_services or []
        })
    
    def get_active_events(
        self, 
        timestamp: datetime
    ) -> List[Dict]:
        """Get all events active at the given timestamp."""
        active = []
        for event in self.events:
            if event["start"] <= timestamp <= event["end"]:
                active.append(event)
        return active
    
    def get_max_expected_impact(
        self, 
        timestamp: datetime
    ) -> float:
        """Get maximum expected impact from active events."""
        active = self.get_active_events(timestamp)
        if not active:
            return 1.0
        return max(event["impact"] for event in active)
    
    @classmethod
    def create_synthetic_calendar(cls, base_time: datetime = None):
        """Create a synthetic event calendar for testing."""
        calendar = cls()
        base = base_time or datetime.now()
        
        # Add various event types
        events = [
            ("Flash Sale", "sale", 0, 2, 3.5),
            ("Marketing Campaign", "marketing", 4, 8, 2.0),
            ("Batch Processing", "batch", 12, 14, 1.8),
            ("Holiday Traffic", "seasonal", 24, 72, 2.5),
        ]
        
        for name, evt_type, start_offset, end_offset, impact in events:
            calendar.add_event(
                name=name,
                event_type=evt_type,
                start_time=base + timedelta(hours=start_offset),
                end_time=base + timedelta(hours=end_offset),
                expected_impact=impact
            )
        
        return calendar

class ContextFeatureExtractor:
    """
    Extract features based on external context.
    """
    
    # Typical high-traffic hours (for time_seasonality)
    HIGH_TRAFFIC_HOURS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    def __init__(self, event_calendar: Optional[EventCalendar] = None):
        self.calendar = event_calendar or EventCalendar()
    
    def extract_event_active(
        self,
        timestamp: datetime = None,
        context: Dict = None
    ) -> float:
        """
        Check if a known event is active.
        
        Returns 1.0 if event is active, 0.0 otherwise.
        Can use either timestamp + calendar or explicit context dict.
        """
        # Use explicit context if provided
        if context and "event_type" in context:
            return 1.0
        
        # Otherwise check calendar
        if timestamp:
            active_events = self.calendar.get_active_events(timestamp)
            return 1.0 if active_events else 0.0
        
        return 0.0
    
    def extract_event_expected_impact(
        self,
        timestamp: datetime = None,
        context: Dict = None
    ) -> float:
        """
        Get expected load multiplier from active events.
        
        Returns normalized value (0-1 scale where 1 = 5x load).
        """
        # Use explicit context
        if context and "load_multiplier" in context:
            impact = context["load_multiplier"]
            return min(1.0, impact / 5.0)  # Normalize
        
        # Check calendar
        if timestamp:
            impact = self.calendar.get_max_expected_impact(timestamp)
            return min(1.0, impact / 5.0)
        
        return 0.0
    
    def extract_time_seasonality(
        self,
        timestamp: datetime = None
    ) -> float:
        """
        Check if current time is a typical high-traffic period.
        
        Returns 1.0 for peak hours, 0.0 for off-peak.
        """
        if timestamp is None:
            return 0.5
        
        hour = timestamp.hour
        day = timestamp.weekday()
        
        # Weekend adjustment
        weekend_factor = 0.8 if day >= 5 else 1.0
        
        # Hour factor
        if hour in self.HIGH_TRAFFIC_HOURS:
            hour_factor = 1.0
        elif hour in [21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
            hour_factor = 0.3
        else:
            hour_factor = 0.6
        
        return hour_factor * weekend_factor
    
    def extract_recent_deployment(
        self,
        timestamp: datetime = None,
        deployment_history: List[datetime] = None,
        window_minutes: int = 30
    ) -> float:
        """
        Check if there was a recent deployment.
        
        Deployments can cause legitimate metric changes.
        """
        if timestamp is None or deployment_history is None:
            return 0.0
        
        window = timedelta(minutes=window_minutes)
        
        for deploy_time in deployment_history:
            if abs((timestamp - deploy_time).total_seconds()) < window.total_seconds():
                return 1.0
        
        return 0.0
    
    def extract_context_confidence(
        self,
        context: Dict = None,
        has_calendar: bool = True
    ) -> float:
        """
        Measure confidence in available context information.
        
        High confidence when we have rich context data.
        Low confidence when context is missing or incomplete.
        """
        confidence = 0.0
        
        if context:
            # Each piece of context adds confidence
            if "event_type" in context:
                confidence += 0.3
            if "load_multiplier" in context:
                confidence += 0.2
            if "event_name" in context:
                confidence += 0.1
            if "expected_error_rate" in context:
                confidence += 0.1
        
        if has_calendar and self.calendar.events:
            confidence += 0.3
        
        return min(1.0, confidence)
    
    def extract_all(
        self,
        timestamp: datetime = None,
        context: Dict = None,
        deployment_history: List[datetime] = None
    ) -> ContextFeatures:
        """Extract all context features."""
        return ContextFeatures(
            event_active=self.extract_event_active(timestamp, context),
            event_expected_impact=self.extract_event_expected_impact(timestamp, context),
            time_seasonality=self.extract_time_seasonality(timestamp),
            recent_deployment=self.extract_recent_deployment(timestamp, deployment_history),
            context_confidence=self.extract_context_confidence(context)
        )

if __name__ == "__main__":
    # Test
    calendar = EventCalendar.create_synthetic_calendar(datetime.now())
    extractor = ContextFeatureExtractor(calendar)
    
    # Test with current time
    features = extractor.extract_all(timestamp=datetime.now())
    print(f"Current context features: {features.to_dict()}")
    
    # Test with explicit context (simulating synthetic load case)
    context = {
        "event_type": "flash_sale",
        "load_multiplier": 3.5,
        "event_name": "Black Friday Sale"
    }
    features = extractor.extract_all(context=context)
    print(f"Event context features: {features.to_dict()}")
```

## Step 2.4: Combined Feature Pipeline

Create file: `src/features/feature_pipeline.py`

```python
"""
Combined feature extraction pipeline.
Orchestrates all feature extractors and produces final feature matrix.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from datetime import datetime

from .workload_features import WorkloadFeatureExtractor, WorkloadFeatures
from .behavioral_features import BehavioralFeatureExtractor, BehavioralFeatures
from .context_features import ContextFeatureExtractor, ContextFeatures, EventCalendar

class FeaturePipeline:
    """
    Main feature extraction pipeline.
    
    Combines workload, behavioral, and context features
    into a single feature vector for classification.
    """
    
    def __init__(
        self,
        event_calendar: EventCalendar = None,
        anomaly_threshold: float = 2.0
    ):
        self.workload_extractor = WorkloadFeatureExtractor()
        self.behavioral_extractor = BehavioralFeatureExtractor(anomaly_threshold)
        self.context_extractor = ContextFeatureExtractor(event_calendar)
        
        # Feature names for consistent ordering
        self.feature_names = [
            # Workload features
            "global_load_ratio",
            "cpu_request_correlation",
            "cross_service_sync",
            "error_rate_delta",
            "latency_cpu_correlation",
            "memory_trend_uniformity",
            # Behavioral features
            "onset_gradient",
            "peak_duration",
            "cascade_score",
            "recovery_indicator",
            "affected_service_ratio",
            "variance_change_ratio",
            # Context features
            "event_active",
            "event_expected_impact",
            "time_seasonality",
            "recent_deployment",
            "context_confidence"
        ]
    
    def extract_features(
        self,
        metrics: pd.DataFrame,
        timestamp: datetime = None,
        context: Dict = None,
        deployment_history: List[datetime] = None
    ) -> Dict[str, float]:
        """
        Extract all features from a single case.
        
        Args:
            metrics: Metrics DataFrame
            timestamp: Timestamp for context lookup
            context: Explicit context dict (for synthetic load cases)
            deployment_history: List of recent deployment times
            
        Returns:
            Dict of feature name -> value
        """
        # Extract each feature group
        workload_feat = self.workload_extractor.extract_all(metrics)
        behavioral_feat = self.behavioral_extractor.extract_all(metrics)
        context_feat = self.context_extractor.extract_all(
            timestamp, context, deployment_history
        )
        
        # Combine into single dict
        features = {}
        features.update(workload_feat.to_dict())
        features.update(behavioral_feat.to_dict())
        features.update(context_feat.to_dict())
        
        return features
    
    def extract_features_batch(
        self,
        cases: List[Union['FailureCase', 'SyntheticLoadCase']],
        labels: List[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract features for multiple cases.
        
        Args:
            cases: List of FailureCase or SyntheticLoadCase objects
            labels: Optional labels for each case
            
        Returns:
            Tuple of (features DataFrame, labels array)
        """
        all_features = []
        case_ids = []
        
        for case in cases:
            # Get context if available (for synthetic load cases)
            context = getattr(case, 'context', None)
            
            features = self.extract_features(
                metrics=case.metrics,
                context=context
            )
            
            all_features.append(features)
            case_ids.append(case.case_id)
        
        # Create DataFrame
        df = pd.DataFrame(all_features, index=case_ids)
        df = df[self.feature_names]  # Ensure consistent column order
        
        # Handle labels
        if labels is None:
            labels = [getattr(case, 'label', 'UNKNOWN') for case in cases]
        
        return df, np.array(labels)
    
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return self.feature_names.copy()
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature names grouped by category."""
        return {
            "workload": [
                "global_load_ratio",
                "cpu_request_correlation",
                "cross_service_sync",
                "error_rate_delta",
                "latency_cpu_correlation",
                "memory_trend_uniformity"
            ],
            "behavioral": [
                "onset_gradient",
                "peak_duration",
                "cascade_score",
                "recovery_indicator",
                "affected_service_ratio",
                "variance_change_ratio"
            ],
            "context": [
                "event_active",
                "event_expected_impact",
                "time_seasonality",
                "recent_deployment",
                "context_confidence"
            ]
        }

def create_feature_matrix(
    cases: List,
    labels: List[str] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience function to create feature matrix.
    
    Args:
        cases: List of case objects
        labels: List of labels
        
    Returns:
        Tuple of (X, y)
    """
    pipeline = FeaturePipeline()
    return pipeline.extract_features_batch(cases, labels)

if __name__ == "__main__":
    # Test the full pipeline
    import sys
    sys.path.insert(0, '..')
    
    from data_loader.rcaeval_loader import load_rcaeval
    from data_loader.synthetic_load_generator import generate_balanced_dataset
    
    # Load fault cases
    fault_cases = load_rcaeval("RE1", "online-boutique")[:10]
    
    # Generate balanced dataset
    all_cases, labels = generate_balanced_dataset(fault_cases, n_load_per_fault=1)
    
    # Extract features
    X, y = create_feature_matrix(all_cases, labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"\nFeature statistics:")
    print(X.describe())
```

---

# PHASE 3: MODEL DEVELOPMENT

## Step 3.1: Anomaly Detector (LSTM Autoencoder Baseline)

Create file: `src/models/anomaly_detector.py`

```python
"""
Anomaly detector using LSTM Autoencoder.
This detects THAT something is anomalous (not WHY).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    
    def __init__(self, data: np.ndarray, seq_length: int = 30):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_length]
        return torch.FloatTensor(seq)

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time series anomaly detection.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_dim, n_features)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        # x: (batch, seq_len, features)
        _, (hidden, _) = self.encoder_lstm(x)
        # hidden: (num_layers, batch, hidden_dim)
        # Use last layer's hidden state
        latent = self.encoder_fc(hidden[-1])
        return latent
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation to sequence."""
        # Expand latent to match sequence length
        hidden = self.decoder_fc(latent)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        output, _ = self.decoder_lstm(hidden)
        reconstruction = self.output_layer(output)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode."""
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction, latent

class AnomalyDetector:
    """
    Wrapper class for anomaly detection.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        seq_length: int = 30,
        threshold_percentile: float = 95
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.threshold_percentile = threshold_percentile
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _preprocess(
        self, 
        metrics: pd.DataFrame,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """Preprocess metrics to numpy array."""
        # Select numeric columns only
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns
        data = metrics[numeric_cols].values
        
        # Handle NaN
        data = np.nan_to_num(data, nan=0.0)
        
        # Scale
        if fit_scaler:
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)
        
        return data
    
    def fit(
        self,
        train_metrics: List[pd.DataFrame],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train the anomaly detector on normal data.
        
        Args:
            train_metrics: List of metrics DataFrames (should be normal data)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Preprocess all training data
        all_data = []
        for metrics in train_metrics:
            data = self._preprocess(metrics, fit_scaler=(len(all_data) == 0))
            all_data.append(data)
        
        combined_data = np.vstack(all_data)
        n_features = combined_data.shape[1]
        
        # Initialize model
        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Create dataset and loader
        dataset = TimeSeriesDataset(combined_data, self.seq_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                reconstruction, _ = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
        
        # Compute threshold from training data
        self._compute_threshold(combined_data)
    
    def _compute_threshold(self, data: np.ndarray):
        """Compute anomaly threshold from reconstruction errors."""
        errors = self.compute_reconstruction_errors(data)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
    
    def compute_reconstruction_errors(
        self, 
        data: np.ndarray
    ) -> np.ndarray:
        """Compute reconstruction error for each time step."""
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(len(data) - self.seq_length):
                seq = torch.FloatTensor(data[i:i+self.seq_length]).unsqueeze(0)
                seq = seq.to(self.device)
                
                reconstruction, _ = self.model(seq)
                error = torch.mean((reconstruction - seq) ** 2).item()
                errors.append(error)
        
        return np.array(errors)
    
    def detect(self, metrics: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Detect anomalies in metrics.
        
        Args:
            metrics: Metrics DataFrame
            
        Returns:
            Tuple of (anomaly_scores, max_score)
        """
        data = self._preprocess(metrics)
        errors = self.compute_reconstruction_errors(data)
        
        # Normalize by threshold
        anomaly_scores = errors / (self.threshold + 1e-10)
        max_score = np.max(anomaly_scores)
        
        return anomaly_scores, max_score
    
    def is_anomalous(self, metrics: pd.DataFrame) -> bool:
        """Check if metrics contain anomaly."""
        _, max_score = self.detect(metrics)
        return max_score > 1.0
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'config': {
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'num_layers': self.num_layers,
                'seq_length': self.seq_length
            }
        }, path)
    
    def load(self, path: str, n_features: int):
        """Load model from file."""
        checkpoint = torch.load(path)
        
        config = checkpoint['config']
        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']

if __name__ == "__main__":
    # Test
    from src.data_loader.rcaeval_loader import load_rcaeval
    
    cases = load_rcaeval("RE1", "online-boutique")[:10]
    
    # Use first few as "normal" training data (simplified for testing)
    train_metrics = [case.metrics for case in cases[:5]]
    
    detector = AnomalyDetector(hidden_dim=32, latent_dim=8, seq_length=20)
    detector.fit(train_metrics, epochs=20)
    
    # Test detection
    for case in cases[5:]:
        scores, max_score = detector.detect(case.metrics)
        print(f"{case.case_id}: max_score={max_score:.3f}, anomalous={max_score > 1.0}")
```

## Step 3.2: Anomaly Classifier

Create file: `src/models/classifier.py`

```python
"""
Anomaly classifier: FAULT vs EXPECTED_LOAD vs UNKNOWN
This is the CORE novel component.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, List, Tuple, Optional

class AnomalyClassifier:
    """
    Classifier to distinguish between:
    - FAULT: Actual system issue
    - EXPECTED_LOAD: Legitimate workload spike
    - UNKNOWN: Unclear, needs investigation
    """
    
    SUPPORTED_MODELS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'mlp': MLPClassifier
    }
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        **model_kwargs
    ):
        """
        Args:
            model_type: Type of classifier
            **model_kwargs: Arguments passed to classifier
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        
        # Set defaults
        if model_type == 'random_forest':
            self.model_kwargs.setdefault('n_estimators', 100)
            self.model_kwargs.setdefault('max_depth', 10)
            self.model_kwargs.setdefault('random_state', 42)
            self.model_kwargs.setdefault('class_weight', 'balanced')
        elif model_type == 'gradient_boosting':
            self.model_kwargs.setdefault('n_estimators', 100)
            self.model_kwargs.setdefault('max_depth', 5)
            self.model_kwargs.setdefault('random_state', 42)
        elif model_type == 'mlp':
            self.model_kwargs.setdefault('hidden_layer_sizes', (64, 32))
            self.model_kwargs.setdefault('max_iter', 500)
            self.model_kwargs.setdefault('random_state', 42)
        
        self.model = self.SUPPORTED_MODELS[model_type](**self.model_kwargs)
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validate: bool = True
    ) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            validate: Whether to run cross-validation
            
        Returns:
            Dict with training results
        """
        self.feature_names = X.columns.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Handle NaN in features
        X_clean = X.fillna(0)
        
        results = {}
        
        # Cross-validation
        if validate:
            cv_scores = cross_val_score(
                self.model, X_clean, y_encoded, 
                cv=5, scoring='f1_weighted'
            )
            results['cv_f1_mean'] = cv_scores.mean()
            results['cv_f1_std'] = cv_scores.std()
            print(f"CV F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Fit on full data
        self.model.fit(X_clean, y_encoded)
        self.is_fitted = True
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            results['feature_importance'] = importance.to_dict()
            
            print("\nTop 10 Important Features:")
            for feat, imp in importance.head(10).items():
                print(f"  {feat}: {imp:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        X_clean = X.fillna(0)
        y_pred_encoded = self.model.predict(X_clean)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with probability for each class
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        X_clean = X.fillna(0)
        probas = self.model.predict_proba(X_clean)
        
        return pd.DataFrame(
            probas,
            columns=self.label_encoder.classes_,
            index=X.index
        )
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence threshold.
        
        Returns UNKNOWN if confidence is below threshold.
        
        Args:
            X: Feature matrix
            confidence_threshold: Minimum probability to make prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probas = self.predict_proba(X)
        
        predictions = []
        confidences = []
        
        for idx, row in probas.iterrows():
            max_prob = row.max()
            max_class = row.idxmax()
            
            if max_prob >= confidence_threshold:
                predictions.append(max_class)
            else:
                predictions.append('UNKNOWN')
            
            confidences.append(max_prob)
        
        return np.array(predictions), np.array(confidences)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X)
        
        results = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred, labels=self.label_encoder.classes_)
        }
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        print("\nConfusion Matrix:")
        print(pd.DataFrame(
            results['confusion_matrix'],
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        ))
        
        return results
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        
        return None
    
    def save(self, path: str):
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyClassifier':
        """Load model from file."""
        data = joblib.load(path)
        
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.label_encoder = data['label_encoder']
        classifier.feature_names = data['feature_names']
        classifier.is_fitted = True
        
        return classifier

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    model_type: str = 'random_forest'
) -> Tuple[AnomalyClassifier, Dict]:
    """
    Convenience function to train and evaluate.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Test set proportion
        model_type: Type of classifier
        
    Returns:
        Tuple of (trained classifier, results dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    
    # Train
    classifier = AnomalyClassifier(model_type=model_type)
    train_results = classifier.fit(X_train, y_train)
    
    # Evaluate
    eval_results = classifier.evaluate(X_test, y_test)
    
    results = {
        'train': train_results,
        'eval': eval_results
    }
    
    return classifier, results

if __name__ == "__main__":
    # Test
    from src.data_loader.rcaeval_loader import load_rcaeval
    from src.data_loader.synthetic_load_generator import generate_balanced_dataset
    from src.features.feature_pipeline import create_feature_matrix
    
    # Load and generate data
    fault_cases = load_rcaeval("RE1", "online-boutique")[:20]
    all_cases, labels = generate_balanced_dataset(fault_cases, n_load_per_fault=1)
    
    # Extract features
    X, y = create_feature_matrix(all_cases, labels)
    
    # Train and evaluate
    classifier, results = train_and_evaluate(X, y, model_type='random_forest')
    
    print(f"\nTest F1: {results['eval']['classification_report']['weighted avg']['f1-score']:.3f}")
```

---

# PHASE 4: EVALUATION & MAIN PIPELINE

## Step 4.1: Evaluation Metrics

Create file: `src/evaluation/metrics.py`

```python
"""
Evaluation metrics specific to anomaly attribution.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List

def compute_false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: str = 'FAULT'
) -> float:
    """
    Compute false positive rate for a specific class.
    
    For our use case: FP = predicting FAULT when actually EXPECTED_LOAD
    """
    mask_actual_negative = y_true != positive_class
    if mask_actual_negative.sum() == 0:
        return 0.0
    
    false_positives = ((y_pred == positive_class) & mask_actual_negative).sum()
    return false_positives / mask_actual_negative.sum()

def compute_false_positive_reduction(
    baseline_fp_rate: float,
    model_fp_rate: float
) -> float:
    """
    Compute FP reduction percentage.
    
    This is the KEY metric for the novel contribution.
    """
    if baseline_fp_rate == 0:
        return 0.0
    return (baseline_fp_rate - model_fp_rate) / baseline_fp_rate

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_fp_rate: float = None
) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        baseline_fp_rate: FP rate of baseline method (for FP reduction)
        
    Returns:
        Dict of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    labels = np.unique(np.concatenate([y_true, y_pred]))
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            metrics[f'precision_{label}'] = precision_score(
                y_true == label, y_pred == label, zero_division=0
            )
            metrics[f'recall_{label}'] = recall_score(
                y_true == label, y_pred == label, zero_division=0
            )
            metrics[f'f1_{label}'] = f1_score(
                y_true == label, y_pred == label, zero_division=0
            )
    
    # FP rate for FAULT class
    fp_rate = compute_false_positive_rate(y_true, y_pred, 'FAULT')
    metrics['fp_rate_FAULT'] = fp_rate
    
    # FP reduction if baseline provided
    if baseline_fp_rate is not None:
        metrics['fp_reduction'] = compute_false_positive_reduction(
            baseline_fp_rate, fp_rate
        )
    
    return metrics

def print_evaluation_summary(metrics: Dict):
    """Print formatted evaluation summary."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  F1 Score:  {metrics['f1_weighted']:.3f}")
    print(f"  Precision: {metrics['precision_weighted']:.3f}")
    print(f"  Recall:    {metrics['recall_weighted']:.3f}")
    
    print(f"\nFalse Positive Analysis:")
    print(f"  FP Rate (FAULT class): {metrics.get('fp_rate_FAULT', 'N/A'):.3f}")
    if 'fp_reduction' in metrics:
        print(f"  FP Reduction vs Baseline: {metrics['fp_reduction']*100:.1f}%")
    
    print("="*50)
```

## Step 4.2: Main Pipeline

Create file: `src/main.py`

```python
"""
Main CEAA Pipeline.
Complete end-to-end workflow for anomaly attribution.
"""
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Local imports
from data_loader.rcaeval_loader import RCAEvalLoader
from data_loader.synthetic_load_generator import generate_balanced_dataset
from features.feature_pipeline import FeaturePipeline
from models.classifier import AnomalyClassifier, train_and_evaluate
from evaluation.metrics import compute_all_metrics, print_evaluation_summary

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_pipeline(
    dataset: str = "RE1",
    system: str = "online-boutique",
    n_load_per_fault: int = 1,
    model_type: str = "random_forest",
    output_dir: str = "outputs/results"
):
    """
    Run the complete CEAA pipeline.
    
    Args:
        dataset: RCAEval dataset (RE1, RE2, RE3)
        system: Target system
        n_load_per_fault: Number of synthetic load cases per fault
        model_type: Classifier type
        output_dir: Output directory
    """
    print("="*60)
    print("CEAA: Context-Enriched Anomaly Attribution")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset}/{system}")
    print(f"  Load cases per fault: {n_load_per_fault}")
    print(f"  Model: {model_type}")
    
    # Step 1: Load fault data
    print("\n[1/5] Loading fault cases from RCAEval...")
    loader = RCAEvalLoader()
    fault_cases = loader.load_dataset(dataset, system)
    print(f"  Loaded {len(fault_cases)} fault cases")
    
    # Step 2: Generate balanced dataset
    print("\n[2/5] Generating synthetic load spike cases...")
    all_cases, labels = generate_balanced_dataset(
        fault_cases, 
        n_load_per_fault=n_load_per_fault
    )
    print(f"  Total cases: {len(all_cases)}")
    print(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Step 3: Extract features
    print("\n[3/5] Extracting features...")
    pipeline = FeaturePipeline()
    X, y = pipeline.extract_features_batch(all_cases, labels)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {list(X.columns)}")
    
    # Step 4: Train and evaluate
    print("\n[4/5] Training classifier...")
    classifier, results = train_and_evaluate(
        X, y, 
        test_size=0.2, 
        model_type=model_type
    )
    
    # Step 5: Compute final metrics
    print("\n[5/5] Computing evaluation metrics...")
    
    # For comparison: baseline assumes all anomalies are faults
    baseline_predictions = np.array(['FAULT'] * len(y))
    baseline_metrics = compute_all_metrics(y, baseline_predictions)
    baseline_fp_rate = baseline_metrics['fp_rate_FAULT']
    
    # Our model's predictions (on test set, already computed in train_and_evaluate)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    y_pred = classifier.predict(X_test)
    
    model_metrics = compute_all_metrics(y_test, y_pred, baseline_fp_rate)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nBaseline (all anomalies = FAULT):")
    print(f"  FP Rate: {baseline_fp_rate:.3f}")
    
    print(f"\nCEAA Model ({model_type}):")
    print(f"  Accuracy: {model_metrics['accuracy']:.3f}")
    print(f"  F1 Score: {model_metrics['f1_weighted']:.3f}")
    print(f"  FP Rate: {model_metrics['fp_rate_FAULT']:.3f}")
    print(f"  FP Reduction: {model_metrics.get('fp_reduction', 0)*100:.1f}%")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_df = pd.DataFrame([model_metrics])
    metrics_df.to_csv(output_path / f"metrics_{timestamp}.csv", index=False)
    
    # Save model
    model_path = output_path / f"classifier_{timestamp}.joblib"
    classifier.save(str(model_path))
    
    # Save feature importance
    importance = classifier.get_feature_importance()
    if importance is not None:
        importance.to_csv(output_path / f"feature_importance_{timestamp}.csv")
    
    print(f"\nResults saved to: {output_path}")
    
    return classifier, model_metrics

def main():
    parser = argparse.ArgumentParser(description="CEAA: Context-Enriched Anomaly Attribution")
    
    parser.add_argument('--dataset', type=str, default='RE1',
                       choices=['RE1', 'RE2', 'RE3'],
                       help='RCAEval dataset to use')
    parser.add_argument('--system', type=str, default='online-boutique',
                       choices=['online-boutique', 'sock-shop', 'train-ticket'],
                       help='Target microservice system')
    parser.add_argument('--load-ratio', type=int, default=1,
                       help='Number of synthetic load cases per fault')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'mlp'],
                       help='Classifier model type')
    parser.add_argument('--output', type=str, default='outputs/results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_pipeline(
        dataset=args.dataset,
        system=args.system,
        n_load_per_fault=args.load_ratio,
        model_type=args.model,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
```

## Step 4.3: Quick Start Script

Create file: `run_experiment.sh`

```bash
#!/bin/bash
# Quick start script for CEAA experiment

set -e

echo "CEAA: Context-Enriched Anomaly Attribution"
echo "==========================================="

# Check Python version
python3 --version

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Download data (if not exists)
if [ ! -d "data/raw/RE1" ]; then
    echo "Downloading RCAEval dataset..."
    python -c "from src.data_loader.download_data import download_rcaeval_dataset; download_rcaeval_dataset('RE1', ['online-boutique'])"
fi

# Run experiment
echo "Running CEAA pipeline..."
python -m src.main \
    --dataset RE1 \
    --system online-boutique \
    --load-ratio 1 \
    --model random_forest \
    --output outputs/results

echo ""
echo "Experiment complete! Check outputs/results/ for results."
```

---

# PHASE 5: SUMMARY & AGENT INSTRUCTIONS

## Complete File Structure

```
ceaa-project/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/           # RCAEval downloads
│   ├── processed/     # Preprocessed data
│   └── synthetic/     # Generated load spikes
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader/
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   ├── rcaeval_loader.py
│   │   └── synthetic_load_generator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── workload_features.py
│   │   ├── behavioral_features.py
│   │   ├── context_features.py
│   │   └── feature_pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py
│   │   └── classifier.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       └── __init__.py
├── models/
│   ├── checkpoints/
│   └── final/
├── outputs/
│   ├── figures/
│   └── results/
├── notebooks/
│   └── exploration.ipynb
├── tests/
├── requirements.txt
├── run_experiment.sh
└── README.md
```

## Agent Execution Order

```
1. Create project structure (Phase 0)
2. Install dependencies (Phase 0)
3. Create configuration (Phase 0)
4. Implement data loader (Phase 1, Steps 1.1-1.3)
5. Implement feature extractors (Phase 2, Steps 2.1-2.4)
6. Implement classifier (Phase 3, Steps 3.1-3.2)
7. Implement evaluation (Phase 4, Step 4.1)
8. Implement main pipeline (Phase 4, Step 4.2)
9. Run experiment (Phase 4, Step 4.3)
```

## Expected Output

After running the complete pipeline, you should see:

```
RESULTS SUMMARY
============================================================

Baseline (all anomalies = FAULT):
  FP Rate: 0.500  (50% of anomalies are actually load spikes)

CEAA Model (random_forest):
  Accuracy: 0.85+
  F1 Score: 0.85+
  FP Rate: 0.15-0.25
  FP Reduction: 50-70%
```

This demonstrates the NOVEL CONTRIBUTION: significant reduction in false positives by correctly classifying load spikes.

---

# SUCCESS CRITERIA

| Metric | Target | Notes |
|--------|--------|-------|
| **FP Reduction** | >40% | Main contribution |
| **Fault Detection Recall** | >90% | Must not miss real faults |
| **Overall Accuracy** | >80% | 3-class classification |
| **Code Quality** | Runnable | All tests pass |
