#!/bin/bash

echo "============================================================================="
echo "                    SETTING UP VIRTUAL ENVIRONMENT"
echo "============================================================================="
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install packages
echo ""
echo "Installing packages (this takes 5-10 minutes)..."
pip install --upgrade pip
pip install numpy pandas scipy torch torchvision tqdm scikit-learn psutil requests matplotlib

echo ""
echo "============================================================================="
echo "                    RUNNING BENCHMARK"
echo "============================================================================="
echo ""

# Run benchmark
python benchmark_m3.py

echo ""
echo "============================================================================="
echo "                    COMPLETE!"
echo "============================================================================="
echo ""
echo "Results saved to: benchmark_results/movielens_100k/"
echo "  - benchmark_report.txt"
echo "  - metrics.json"
echo "  - plots/metrics_comparison.png"
echo ""

