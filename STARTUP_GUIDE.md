# 🚀 ML/DL Trainer - Application Startup Guide

## Quick Start

### Option 1: Using Python Script (Recommended)
```bash
cd c:\Users\rudra\Downloads\ML_DL_Trainer
python run_app.py
```

### Option 2: Using Batch File (Windows)
```bash
cd c:\Users\rudra\Downloads\ML_DL_Trainer
run_app.bat
```

### Option 3: Direct Streamlit Command
```bash
cd c:\Users\rudra\Downloads\ML_DL_Trainer
streamlit run app/main.py
```

---

## 🌐 Access the Application

Once started, open your browser and go to:
```
http://localhost:8501
```

---

## 📋 Application Features

### 1. **Home Page** 🏠
- Overview of the platform
- Quick start guide
- Supported models list
- Platform statistics

### 2. **Data Upload** 📤
- Upload CSV files
- Load sample datasets (Iris, Wine)
- View data preview and statistics
- Automatic data validation

### 3. **EDA / Data Understanding** 📊
- **Overview Tab**: Dataset summary, column info, statistics
- **Missing Values Tab**: Analysis and visualization
- **Target Analysis Tab**: Classification/Regression detection
- **Feature Analysis Tab**: Numerical and categorical features
- **Correlation Tab**: Feature-target relationships

### 4. **Training** ⚙️
- Select task type (Classification/Regression)
- Choose model algorithm
- Configure hyperparameters
- Train model with real-time feedback

### 5. **Results** 📈
- View performance metrics
- Download trained model (PKL format)
- Download metrics (JSON format)
- Model information display

### 6. **About** ℹ️
- Platform information
- Supported algorithms
- Architecture details
- Acknowledgments

---

## 🤖 Supported Models

### Machine Learning (Scikit-learn)

**Classification:**
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

**Regression:**
- Linear Regression
- Random Forest
- Support Vector Regression (SVR)
- Gradient Boosting

### Deep Learning (TensorFlow/Keras)
- Sequential Neural Networks
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN/LSTM)

---

## 📊 Workflow Example

### Step 1: Upload Data
1. Go to **Data Upload** tab
2. Click "Load Iris Dataset" (or upload your CSV)
3. View data preview and statistics

### Step 2: Explore Data
1. Go to **EDA / Data Understanding** tab
2. Review data quality warnings
3. Analyze target variable
4. Check feature distributions
5. Examine correlations

### Step 3: Train Model
1. Go to **Training** tab
2. Select "Classification" task
3. Choose "Random Forest" algorithm
4. Adjust hyperparameters if needed
5. Click "Start Training"

### Step 4: View Results
1. Go to **Results** tab
2. View performance metrics
3. Download trained model (PKL)
4. Download metrics (JSON)

---

## 🔧 System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for dependencies
- **Browser**: Chrome, Firefox, Safari, or Edge

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
plotly>=5.17.0
matplotlib>=3.7.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Issue: Port 8501 already in use
**Solution**: Use a different port
```bash
streamlit run app/main.py --server.port 8502
```

### Issue: Module not found errors
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Application crashes on EDA
**Solution**: Ensure data has valid target column
- Check for missing values in target
- Ensure target column is selected
- Try with sample dataset first

### Issue: Model training fails
**Solution**: Check data quality
- Remove rows with missing values
- Ensure target has at least 2 unique values (classification)
- Check for non-numeric features

---

## 📝 Configuration

Edit `app/config.py` to customize:

```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # Max upload size
DEFAULT_TEST_SIZE = 0.2             # Train-test split
DEFAULT_CV_FOLDS = 5                # Cross-validation folds
DEFAULT_EPOCHS = 50                 # DL epochs
DEFAULT_BATCH_SIZE = 32             # DL batch size
```

---

## 🎯 Tips & Tricks

### For Best Performance:
1. Use datasets with 1000-100,000 rows
2. Start with sample datasets to learn
3. Use Random Forest for general-purpose tasks
4. Enable cross-validation for robust evaluation
5. Download models for later use

### For Data Exploration:
1. Always run EDA before training
2. Check data quality warnings
3. Analyze target distribution
4. Review feature correlations
5. Look for missing values

### For Model Training:
1. Start with default hyperparameters
2. Use stratified split for imbalanced data
3. Compare multiple models
4. Download best model for production
5. Save metrics for documentation

---

## 📊 Sample Datasets

### Iris Dataset
- **Rows**: 150
- **Features**: 4 (sepal length, width, petal length, width)
- **Target**: 3 classes (setosa, versicolor, virginica)
- **Task**: Classification

### Wine Dataset
- **Rows**: 178
- **Features**: 13 (alcohol, acidity, color, etc.)
- **Target**: 3 classes (wine cultivars)
- **Task**: Classification

---

## 🔐 Security Notes

- ✅ No data is stored permanently
- ✅ Models are saved locally only
- ✅ No external API calls
- ✅ All processing is local
- ⚠️ Not suitable for sensitive data without authentication

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the README.md file
3. Check application logs in `logs/app.log`
4. Verify all dependencies are installed

---

## 🎓 Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Scikit-learn Docs**: https://scikit-learn.org
- **TensorFlow Docs**: https://www.tensorflow.org
- **Pandas Docs**: https://pandas.pydata.org

---

## ✅ Verification Checklist

Before running the application, ensure:

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated (if using one)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Project directory accessible
- [ ] Port 8501 is available
- [ ] Sufficient disk space (500MB+)
- [ ] Internet connection (for initial setup)

---

## 🚀 Ready to Start?

Run the application now:

```bash
python run_app.py
```

Then open: **http://localhost:8501**

Enjoy training your ML/DL models! 🎉

