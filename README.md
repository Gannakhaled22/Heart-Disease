
# Heart Disease Prediction

This repository contains a Jupyter/Colab notebook (HEART.ipynb) that explores a heart disease dataset and trains several models to predict the presence of heart disease. The notebook includes preprocessing, model training, evaluation, and a small deep learning model.

## Files
- `HEART.ipynb` — main notebook with the full analysis and modeling pipeline.
- `Heart Disease Dataset.csv` — dataset used by the notebook (not included here).

## Summary of the workflow
1. Load dataset: `Heart Disease Dataset.csv`.
2. Quick EDA: shape, types, value counts and descriptive statistics.
3. Prepare features:
   - Separate X / y (target = `HeartDisease`).
   - Identify categorical columns and numerical columns.
   - One-hot encode categorical columns with `pd.get_dummies(..., drop_first=True)`.
   - Standard-scale numerical columns with `StandardScaler`.
4. Train/test split: 80/20 with `stratify=y`.
5. Models trained and evaluated:
   - Logistic Regression (sklearn)
   - Random Forest (sklearn)
   - Neural Network (TensorFlow / Keras)
6. Evaluation metrics shown: accuracy, classification report (precision/recall/F1), confusion matrix, and feature importances (Random Forest).

## Key results (from the notebook)
- Logistic Regression Accuracy: ~0.8859
- Random Forest Accuracy: ~0.8750
- Neural Network Accuracy: ~0.8804

Top features by feature importance (Random Forest):
- `ST_Slope_Up`
- `MaxHR`
- `ST_Slope_Flat`
- `Cholesterol`
- `Oldpeak`
- `ExerciseAngina_Y`
- `Age`
- `RestingBP`

Interpretation: Logistic Regression and the Neural Network achieved similar best performance; Random Forest was slightly lower in the demonstrated runs. The ST slope, max heart rate, cholesterol and oldpeak (ST depression) were among the most important features.

## Requirements
Install the packages below (example using pip):

pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Notes:
- Use a Python 3.8+ environment for best compatibility.
- If running on Colab, TensorFlow is preinstalled on many runtimes; adjust versions if needed.

## How to run
- Option 1 — Colab:
  1. Upload `HEART.ipynb` and `Heart Disease Dataset.csv` to Google Drive or open the notebook in Colab.
  2. Ensure the dataset path in the notebook matches (e.g. upload the CSV to the session or mount Drive).
  3. Run the notebook cells top-to-bottom.

- Option 2 — Local:
  1. Create a virtual environment and install requirements.
  2. Place `Heart Disease Dataset.csv` in the same folder as the notebook.
  3. Open `HEART.ipynb` with JupyterLab / Jupyter Notebook and run cells.

## Reproducibility tips
- Set `random_state` where applicable (the notebook uses `random_state=42` in model training and splitting).
- When experimenting with the neural network, results can vary slightly due to nondeterministic GPU operations — set seeds and use CPU or deterministic flags if strict reproducibility is required.
- Save the fitted scalers and encoders if you intend to deploy the model (e.g. with `joblib`).

## Next steps / improvements
- Perform more thorough hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for RF and logistic regression.
- Use cross-validation for more robust performance estimates.
- Handle potential class imbalance (though in this dataset the classes are fairly balanced).
- Add calibration for probability outputs (if probabilities will be used in decision-making).
- Try additional models (XGBoost, LightGBM) and ensembling.
- Add model explainability (SHAP, LIME) to better interpret predictions.

## License
Specify your preferred license (e.g., MIT) if you plan to share this repository publicly.

## Contact
If you want help extending the notebook or converting it to a script/serving pipeline, open an issue or contact the author.
