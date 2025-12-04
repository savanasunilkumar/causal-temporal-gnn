# Causal Temporal GNN (UACT-GNN)

A production-ready recommendation system optimized for large-scale datasets (100M-1B+ interactions). This system combines causal discovery, temporal modeling, multi-modal learning, and graph neural networks using PyTorch Geometric.

## Features

### Core Capabilities
- **PyTorch Geometric Integration**: Uses industry-standard GNN framework with optimized message passing.
- **Causal Discovery**: Implements Granger causality and PC algorithm to discover causal relationships.
- **Temporal Modeling**: Captures temporal dynamics using attention layers and transformers.
- **Multi-Modal Learning**: Processes text, images, numeric, and categorical features.
- **Sparse Graph Operations**: Efficient sparse tensors for memory reduction.
- **Neighbor Sampling**: Handles graphs that do not fit in GPU memory.
- **Zero-Shot Cold Start**: Handles new users/items using pretrained models and learnable fusion.
- **Universal Data Processing**: Automatically detects and processes various data formats (CSV, JSON, Parquet).

### Production Features
- **Optimized Message Passing**: C++/CUDA kernels for improved speed.
- **Gradient Checkpointing**: Memory reduction for deep models.
- **Causal Graph Caching**: Precompute and load causal graphs efficiently.
- **Distributed Training**: Multi-GPU support with PyTorch DDP.
- **Mixed Precision Training**: FP16/BF16 support.
- **Model Checkpointing**: Automatic save/resume functionality.
- **Experiment Logging**: Support for Weights & Biases and TensorBoard.
- **GPU Profiling**: Monitor memory and performance bottlenecks.
- **Comprehensive Evaluation**: Precision, Recall, NDCG, Hit Ratio metrics.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd CausalGNN

# Install core dependencies
pip install -r requirements.txt

# For CUDA 11.8 (recommended)
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu118.html
```

### Optional Dependencies

For advanced features, install optional dependencies:

```bash
# For faster PyG operations
pip install pyg-lib

# For advanced causal discovery
pip install causal-learn

# For zero-shot text features
pip install transformers

# For zero-shot image features
pip install opencv-python Pillow

# For experiment logging
pip install tensorboard wandb
```

## Quick Start

### 1. Prepare Your Data

The system automatically detects data format and schema. Supported formats:
- CSV
- JSON
- Parquet

Example data structure:
```csv
userId,movieId,rating,timestamp
1,123,5,1609459200
1,456,4,1609545600
2,123,3,1609632000
```

### 2. Create a Sample Dataset

```bash
python causal_gnn/scripts/preprocess.py --create_sample
```

### 3. Train the Model

```bash
# Basic training
python causal_gnn/scripts/train.py \
    --data_path ./data/interactions.csv \
    --embedding_dim 64 \
    --num_layers 3 \
    --num_epochs 20 \
    --batch_size 1024 \
    --use_amp

# For large datasets (100M+ interactions)
python causal_gnn/scripts/train.py \
    --data_path ./data/heavy_interactions.csv \
    --embedding_dim 128 \
    --batch_size 4096 \
    --use_gradient_checkpointing \
    --use_neighbor_sampling \
    --use_amp
```

### 4. Evaluate the Model

```bash
python causal_gnn/scripts/evaluate.py \
    --model_path ./output/final_model.pt \
    --data_path ./data/interactions.csv
```

## Usage

### Python API

```python
from causal_gnn.config import Config
from causal_gnn.training.trainer import RecommendationSystem

# Create configuration
config = Config(
    embedding_dim=64,
    num_layers=3,
    time_steps=10,
    learning_rate=0.001,
    batch_size=1024,
    num_epochs=20,
    causal_method='advanced',
    use_amp=True
)

# Initialize system
rec_system = RecommendationSystem(config)

# Load and prepare data
rec_system.load_data('./data/interactions.csv')
rec_system.preprocess_data()
rec_system.split_data()
rec_system.create_graph()

# Train model
rec_system.initialize_model()
history = rec_system.train()

# Evaluate
metrics = rec_system.evaluate('test', k_values=[5, 10, 20])

# Generate recommendations
recommendations, scores = rec_system.generate_recommendations(user_id=1, top_k=10)
```

## Benchmarking

For testing on Apple M3 with 8GB memory, run the benchmark suite:

```bash
python benchmark_m3.py
```

**Features:**
- Auto-downloads MovieLens 100K dataset
- Benchmarks 5 models: Popular, BPR, NCF, LightGCN, CausalGNN
- Optimized for Apple Silicon (MPS acceleration)
- Saves comprehensive results and plots

### Distributed Training

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 causal_gnn/scripts/train.py \
    --data_path ./data/interactions.csv \
    --distributed \
    --use_amp
```

### Cold Start Recommendations

```python
# For new users
cold_start_user = {
    'age': 25,
    'gender': 'F',
    'occupation': 'student'
}

recommendations, scores = rec_system.generate_cold_start_recommendations(
    cold_start_user, top_k=10
)
```

## Configuration

Key configuration parameters:

### Model Architecture
- `embedding_dim`: Dimension of embeddings (default: 64)
- `num_layers`: Number of GNN layers (default: 3)
- `time_steps`: Number of temporal time steps (default: 10)
- `dropout`: Dropout rate (default: 0.1)
- `causal_strength`: Weight of causal connections (default: 0.5)
- `causal_method`: 'simple' or 'advanced' (default: 'advanced')

### Training
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 1024)
- `num_epochs`: Number of epochs (default: 20)
- `neg_samples`: Negative samples per positive (default: 1)
- `weight_decay`: L2 regularization (default: 0.0001)
- `early_stopping_patience`: Epochs to wait before stopping (default: 5)

### System
- `device`: 'cuda' or 'cpu'
- `use_amp`: Enable mixed precision training
- `distributed`: Enable distributed training
- `use_wandb`: Enable Weights & Biases logging
- `use_tensorboard`: Enable TensorBoard logging

## Architecture

```
causal_gnn/
├── __init__.py
├── config.py              # Configuration management
├── models/                # Model components
│   ├── uact_gnn.py       # Main UACT-GNN model
│   └── fusion.py         # Multi-modal fusion
├── data/                  # Data processing
│   ├── processor.py      # Universal data processor
│   ├── dataset.py        # PyTorch datasets
│   └── samplers.py       # Negative sampling
├── causal/               # Causal discovery
│   └── discovery.py      # Granger causality, PC algorithm
├── training/             # Training components
│   ├── trainer.py        # Main training loop
│   └── evaluator.py      # Evaluation metrics
├── utils/                # Utilities
│   ├── cold_start.py     # Zero-shot cold start
│   ├── checkpointing.py  # Model checkpointing
│   └── logging.py        # Experiment logging
└── scripts/              # Executable scripts
    ├── preprocess.py     # Data preprocessing
    ├── train.py          # Training script
    └── evaluate.py       # Evaluation script
```

## Datasets

The system is designed to handle large-scale datasets:

### Tested Datasets
- MovieLens-25M (~25M ratings, 162K users, 62K movies)
- Amazon Reviews (~233M ratings, 43M users, 3M products)
- Custom datasets with millions of interactions

### Dataset Format
Minimum required columns:
- User ID column (auto-detected)
- Item ID column (auto-detected)
- Interaction/rating column (optional)
- Timestamp column (optional, but recommended)

## Performance Optimizations

### For Large Datasets (100M+ interactions)

1. **Enable Mixed Precision**: `use_amp=True` (2-3x memory reduction)
2. **Use Distributed Training**: Multi-GPU support via PyTorch DDP
3. **Increase Batch Size**: Utilize gradient accumulation if needed
4. **Precompute Causal Graphs**: Avoid computing on every forward pass
5. **GPU-Accelerated Sampling**: Use vectorized negative sampling

### Memory Optimization
- Gradient checkpointing (for large models)
- Sparse tensor representations for graphs
- CPU offloading for large embeddings

### Speed Optimization
- DataLoader with `num_workers` and `pin_memory`
- Efficient negative sampling on GPU
- Causal graph caching

## Citation

If you use this code in your research, please cite:

```bibtex
@software{uact_gnn_2024,
  title={Causal Temporal GNN},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/CausalGNN}
}
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and documentation
- Contact: [your-email@example.com]

## Acknowledgments

This implementation builds upon research in:
- Graph Neural Networks
- Causal Discovery
- Temporal Modeling
- Multi-Modal Learning
- Recommendation Systems
