# Modified GMOCAT: Coverage-Enhanced Computerized Adaptive Testing

**Master's Thesis Project**  
**Author:** Yasa Khalqiah  
**Institution:** Gadjah Mada University  
**Year:** 2026

## Research Overview

This repository contains the implementation of **Modified GMOCAT** - an enhanced Computerized Adaptive Testing (CAT) system that significantly improves knowledge concept coverage through three key modifications:

### Key Modifications:

1. **Coverage-Aware Reward**  
   Deficit-based penalty scaling: `reward × (1 + deficit)` to prioritize uncovered knowledge concepts

2. **Uncertainty-Based Termination (UBT)**  
   MC Dropout-based early stopping mechanism to optimize test length while maintaining prediction quality

3. **Adaptive Diversity Weight**  
   Dynamic diversity bonus for question selection to prevent overfitting and improve generalization

## Main Results

### Overall Performance (Seed 42, 50 Epochs):
- **Baseline (Wang et al., 2023):** COV@20 = 0.5130
- **Modified GMOCAT:** COV@20 = **0.6308**
- **Improvement:** **+23.0%**

### Ablation Study Results:

| Modification | COV@20 | AUC@20 | ACC@20 | Contribution |
|--------------|--------|--------|--------|--------------|
| **Full Model** | **0.6308** | 0.6312 | 0.7525 | Baseline |
| Coverage-Aware Reward only | 0.6131 | 0.5946 | 0.7427 | **97.2%** (Dominant) |
| UBT only | 0.5175 | 0.6178 | 0.7418 | 82.0% |
| Adaptive Diversity only | 0.4405 | 0.6170 | 0.7468 | 69.8% |

**Synergy Effect:** +2.8% gain from combining all three modifications

## Key Findings

1. **Coverage-Aware Reward** is the dominant contributor (97.2% retention), driving fast and consistent coverage growth
2. **UBT** provides highest AUC (0.6178) through uncertainty-based quality focus, though minimal test length reduction
3. **Adaptive Diversity** maintains high accuracy despite lowest coverage - supporting role for generalization
4. **Temporal Analysis** reveals fundamental trade-off: Coverage strategy (breadth) vs UBT strategy (prediction quality)

## Project Structure
cd ~/thesis-backup

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Environments
venv/
ENV/
env/

# Model checkpoints (too large for GitHub)
*.pt
*.pth
*.pkl
models/
checkpoints/
baseline_log/
pretrain_log/

# Large log files (except results/)
*.log
!results/*.log

# Graph data (too large)
graph_data/
raw_data/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
*.bak

# Temporary files
*.tmp
nohup.out
