# DeepFlow Stamatics — Major Assignment 1

## Bulldozer Sale Price Prediction

Predicted heavy equipment sale prices using a Random Forest Regressor on tabular data (~30k training samples).

### Key Steps
1. Handled null values by replacing with median of existing values
2. Converted object columns to pandas category codes for model compatibility
3. Applied log transform on target since evaluation metric is RMSLE
4. Trained a Random Forest Regressor and generated predictions on test set

### Tech Stack
Python · pandas · NumPy · scikit-learn · Google Colab
