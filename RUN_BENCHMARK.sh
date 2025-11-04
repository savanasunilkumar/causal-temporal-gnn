#!/bin/bash

# M3 Benchmark Runner Script
# This script installs dependencies and runs the benchmark

echo "============================================================================="
echo "                    M3 BENCHMARK - INSTALLATION & RUNNER"
echo "============================================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Install dependencies
echo "Installing dependencies (this may take 5-10 minutes)..."
echo "Installing: numpy, pandas, scipy, torch, psutil, requests, matplotlib, scikit-learn, tqdm"
echo ""

pip3 install numpy pandas scipy torch torchvision psutil requests matplotlib scikit-learn tqdm

echo ""
echo "============================================================================="
echo "                    DEPENDENCIES INSTALLED"
echo "============================================================================="
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy, pandas, torch, psutil, requests, matplotlib, scipy, sklearn, tqdm; print('✓ All dependencies installed successfully!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================="
    echo "                    RUNNING BENCHMARK"
    echo "============================================================================="
    echo ""
    echo "This will take approximately 20-30 minutes on M3 8GB"
    echo "Results will be saved to: benchmark_results/movielens_100k/"
    echo ""
    echo "Press Ctrl+C to stop if needed."
    echo ""
    
    # Run benchmark
    python3 benchmark_m3.py
    
    echo ""
    echo "============================================================================="
    echo "                    BENCHMARK COMPLETE!"
    echo "============================================================================="
    echo ""
    echo "Results saved to:"
    echo "  - benchmark_results/movielens_100k/benchmark_report.txt"
    echo "  - benchmark_results/movielens_100k/metrics.json"
    echo "  - benchmark_results/movielens_100k/plots/metrics_comparison.png"
    echo ""
    echo "Open benchmark_report.txt to see the comparison!"
    echo "============================================================================="
else
    echo ""
    echo "✗ Installation failed. Please install dependencies manually:"
    echo "  pip3 install numpy pandas scipy torch psutil requests matplotlib scikit-learn tqdm"
    echo ""
fi

