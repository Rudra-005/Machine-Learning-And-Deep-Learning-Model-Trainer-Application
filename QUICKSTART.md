# ML/DL Trainer - Quick Start Guide

## Installation & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Application
```bash
streamlit run app/main.py
```

Open browser at: `http://localhost:8501`

### 3. Using Docker (Optional)
```bash
docker-compose up -d
```

## Quick Usage

1. **Upload Data**
   - Go to "Data Upload" tab
   - Select CSV file (max 500MB)
   - Review data quality metrics

2. **Train Model**
   - Choose task: Classification/Regression
   - Select model: ML (SKL) or DL (TensorFlow)
   - Configure hyperparameters
   - Click "Start Training"

3. **View Results**
   - Check metrics, visualizations
   - Download trained model
   - Export results

## File Structure
- `app/` - Streamlit frontend
- `core/` - Data preprocessing & validation
- `models/` - ML/DL model implementations
- `evaluation/` - Metrics & visualizations
- `storage/` - Model & result persistence
- `data/` - Datasets and models storage

## Example Dataset

Create `sample_data.csv`:
```csv
age,income,employed
25,30000,1
35,50000,1
45,80000,1
22,25000,0
55,120000,1
```

## Support
- Issues: Check README.md
- Logs: `logs/app.log`
- Config: `app/config.py`

## Next Steps
- Explore different models
- Tune hyperparameters
- Compare results
- Deploy to cloud
