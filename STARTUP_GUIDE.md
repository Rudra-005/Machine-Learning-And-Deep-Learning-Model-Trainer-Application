# 🚀 AutoML Mode - Startup Guide

## Quick Start (2 Minutes)

### Step 1: Install Dependencies

```bash
pip install streamlit scikit-learn pandas numpy tensorflow
```

### Step 2: Run the Demo Application

```bash
cd c:\Users\rudra\Downloads\ML_DL_Trainer
streamlit run app_demo.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Using the Application

### Page 1: 📊 Data Loading
1. Click "Load Sample Dataset"
2. Choose "Iris (Classification)" or "Diabetes (Regression)"
3. Click "Load Sample Dataset" button
4. Data is now ready for training

### Page 2: 🧠 AutoML Training
1. Select task type (Classification or Regression)
2. Select a model from the dropdown
3. **AutoML automatically detects the model type**
4. **AutoML automatically selects the training strategy**
5. **UI shows only relevant parameters**
6. Configure parameters (optional)
7. Click "🚀 Start AutoML Training"
8. View results

### Page 3: 📈 Strategy Guide
Learn how AutoML selects strategies for different model types

### Page 4: ℹ️ About
Learn about AutoML Mode features and benefits

---

## 🎯 What to Try

### Try 1: Random Forest (Tree-Based)
1. Load Iris dataset
2. Select "Random Forest"
3. Notice: CV Folds shown, Epochs hidden
4. Train and see K-Fold CV results

### Try 2: Logistic Regression (Iterative)
1. Load Iris dataset
2. Select "Logistic Regression"
3. Notice: CV Folds AND Max Iter shown
4. Train and see K-Fold CV results

### Try 3: SVM (SVM)
1. Load Iris dataset
2. Select "SVM"
3. Notice: CV Folds shown, Max Iter hidden
4. Enable HP Tuning for best results
5. Train and see best hyperparameters

### Try 4: Ridge (Regression)
1. Load Diabetes dataset
2. Select "Ridge"
3. Train and see regression results

---

## 🔍 What to Observe

### AutoML Detection
- Each model automatically categorized
- Strategy selected based on category
- Parameters shown/hidden accordingly

### Parameter Visibility
- **Tree-Based**: CV Folds, HP Tuning
- **Iterative**: CV Folds, Max Iter, HP Tuning
- **SVM**: CV Folds, HP Tuning
- **DL**: Epochs, Batch Size, Learning Rate

### Results Display
- **ML Models**: CV Score ± Std Dev, Test Score
- **DL Models**: Train Loss, Val Loss, Test Accuracy

---

## 📊 Example Output

### Random Forest Results
```
CV Score: 0.9533 ± 0.0245
Test Score: 0.9667
Best Params: {'n_estimators': 100, 'max_depth': 10, ...}
```

### Logistic Regression Results
```
CV Score: 0.9200 ± 0.0356
Test Score: 0.9333
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'models'"
**Solution**: Make sure you're running from the ML_DL_Trainer directory

```bash
cd c:\Users\rudra\Downloads\ML_DL_Trainer
streamlit run app_demo.py
```

### Issue: "No data loaded"
**Solution**: Go to "📊 Data Loading" page and load a sample dataset first

### Issue: "Training failed"
**Solution**: Check the error message and ensure all dependencies are installed

```bash
pip install --upgrade scikit-learn pandas numpy streamlit
```

---

## 📚 Documentation Files

After running the app, read these files to understand the system:

1. **AUTOML_QUICK_REFERENCE.md** - Quick reference guide
2. **AUTOML_DOCUMENTATION.md** - Comprehensive guide
3. **AUTOML_VISUAL_REFERENCE.md** - Visual diagrams
4. **TRAINING_STRATEGY.md** - Strategy explanations

---

## 🎓 Learning Path

### Beginner (5 minutes)
1. Run the app
2. Load Iris dataset
3. Try Random Forest
4. Try Logistic Regression
5. Observe parameter differences

### Intermediate (15 minutes)
1. Try all model types
2. Enable HP tuning
3. Compare results
4. Read AUTOML_QUICK_REFERENCE.md

### Advanced (30 minutes)
1. Read AUTOML_DOCUMENTATION.md
2. Study the source code
3. Understand design patterns
4. Plan extensions

---

## 🚀 Next Steps

### After Running the Demo

1. **Explore the Code**
   - `models/automl.py` - Model detection
   - `models/automl_trainer.py` - Training logic
   - `app/utils/automl_ui.py` - UI components

2. **Read Documentation**
   - Start with AUTOML_QUICK_REFERENCE.md
   - Then read AUTOML_DOCUMENTATION.md
   - Review AUTOML_VISUAL_REFERENCE.md

3. **Integrate into Your Project**
   - Follow AUTOML_INTEGRATION_GUIDE.md
   - Copy files to your project
   - Update your main app

4. **Extend the System**
   - Add new models to MODEL_REGISTRY
   - Add hyperparameter distributions
   - Create custom strategies

---

## 💡 Tips

### Tip 1: Use HP Tuning for Better Results
Enable "Enable Hyperparameter Tuning" for more accurate models

### Tip 2: Increase CV Folds for Robustness
Use 10 folds instead of 5 for more robust evaluation

### Tip 3: Compare Models
Try different models on the same dataset to see which performs best

### Tip 4: Check Strategy Explanation
Click "Why this strategy?" to understand why each strategy was chosen

---

## 📞 Support

### For Questions About...

**How to run**: See "Quick Start" section above  
**How to use**: See "Using the Application" section  
**What to try**: See "What to Try" section  
**Troubleshooting**: See "Troubleshooting" section  
**Understanding**: Read the documentation files  

---

## ✅ Verification Checklist

After running the app, verify:

- ✅ App opens in browser
- ✅ Can load sample dataset
- ✅ Can select models
- ✅ Parameters change based on model
- ✅ Can train models
- ✅ Results display correctly
- ✅ Strategy explanation shows
- ✅ No errors in console

---

## 🎉 You're Ready!

The AutoML Mode application is ready to use. Start with the Quick Start section above and explore the system.

**Enjoy training models with AutoML! 🚀**

---

**Last Updated**: 2026-01-19  
**Version**: 1.0  
**Status**: Ready to Run
