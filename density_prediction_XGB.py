# -*- coding: utf-8 -*-
"""
Machine Learning Regression Model -- XGBoost

This script executes learning and prediction of melting point using XGBoost
Dataset: Bradley Melting Point Dataset ("melt_temp_data.csv" ---RDkit--> "melt_temp_desc.csv")
         <Please run the file named "descriptor_mp.py" before executing the present script>
Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, XGBoost
Learning Algorithm: XGBoost Regressor

Author:
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

### configure the number of available CPU
import os as os
cpu_number = os.cpu_count()
n_jobs = cpu_number - 2

### import some standard libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

### load and define dataframe for the learning dataset
Learning = pd.read_csv("C:/Users/GKlab/Documents/PENELITIAN ZAKIAH/3rd research _ ok/density prediction/ML/XGB/descriptor_dataset_density.csv") # this contains 3025 data points

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

### define X and Y out of the learning data (X: features, Y: values)
X = Learning.drop('density', axis=1)
Y = Learning['density']

print('\n')
print('Features (X): \n')
print(f'Filetype: {type(X)}, Shape: {X.shape}')
print(X)

print('\n')
print('Label (Y): \n')
print(f'Filetype: {type(Y)}, Shape: {Y.shape}')
print(Y)

### split the learning data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)

### train and evaluate the model
from xgboost import XGBRegressor
XGBrgs = XGBRegressor(random_state=1)
from sklearn.model_selection import cross_val_score
cross_val = 10
scores = cross_val_score(XGBrgs, X_train, Y_train, scoring="r2", cv=cross_val, n_jobs=n_jobs)
def display_score(scores):
    print('\n')
    print('Preliminary run: \n')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_score(scores)

### fine tune the model using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [100, 300, 500],
      'max_depth': [10, 50, 100],
      'colsample_bytree': [0.0625, 0.125, 0.25, 0.5, 1]}
    ]
grid_search = GridSearchCV(XGBrgs, param_grid, scoring="r2", cv=cross_val, n_jobs=n_jobs)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

cvres = grid_search.cv_results_

print('\n')
print('Hyperparameter tuning: \n')
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

grid_search.best_estimator_

### re-train the model with the best hyperparameters and the whole training set
XGBrgs_opt = grid_search.best_estimator_
model = XGBrgs_opt.fit(X_train, Y_train)

### analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

cv_pred = cross_val_predict(XGBrgs_opt, X_train, Y_train, cv=cross_val, n_jobs=n_jobs)
R2_cv = r2_score(Y_train, cv_pred)
RMSE_cv = np.sqrt(mean_squared_error(Y_train, cv_pred))

print('\n')
print('Quality assessment with cross-validation (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_cv)
print('RMSE score', RMSE_cv)

### plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, cv_pred, 50, 'tab:blue', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured density (kg/m3)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted density (kg/m3)', labelpad=10, fontproperties=fonts)
import matplotlib.ticker as mticker
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
dpi_assign = 500
plt.savefig('xgb_density_TrainingSet-CV.png', dpi=dpi_assign, bbox_inches='tight')

### analyze and visualize the optimized model performance on TRAINING SET via a SINGLE RUN
train_pred = model.predict(X_train)
R2_train = r2_score(Y_train, train_pred)
RMSE_train = np.sqrt(mean_squared_error(Y_train, train_pred))

# Calculate ARDD, MAPE, MAE
ARD_D = np.mean(np.abs((Y_train - cv_pred) / Y_train))  # Average Relative Deviation
MAPE = np.mean(np.abs((Y_train - cv_pred) / Y_train)) * 100  # Mean Absolute Percentage Error
MAE = np.mean(np.abs(Y_train - cv_pred))  # Mean Absolute Error

print('\n')
print('Quality assessment with cross-validation (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_train)
print('RMSE score: ', RMSE_train)
print('ARDD: ', ARD_D)
print('MAPE: ', MAPE)
print('MAE: ', MAE)



### analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

cv_pred = cross_val_predict(XGBrgs_opt, X_train, Y_train, cv=cross_val, n_jobs=n_jobs)
R2_cv = r2_score(Y_train, cv_pred)
RMSE_cv = np.sqrt(mean_squared_error(Y_train, cv_pred))




### plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, train_pred, 50, 'tab:grey', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured density (kg/m3)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted density (kg/m3)', labelpad=10, fontproperties=fonts)
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
dpi_assign = 500
plt.savefig('xgb_density_TrainingSet.png', dpi=dpi_assign, bbox_inches='tight')

### analyze and visualize the optimized model performance on TEST SET via a SINGLE RUN
test_pred = model.predict(X_test)
R2_test = r2_score(Y_test, test_pred)
RMSE_test = np.sqrt(mean_squared_error(Y_test, test_pred))

print('\n')
print('Learning results for test set (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_test)
print('RMSE score: ', RMSE_test)

### plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_test, test_pred, 50, 'tab:red', alpha=0.2)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=2)
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured density (kg/m3)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted density (kg/m3)', labelpad=10, fontproperties=fonts)
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
dpi_assign = 500
plt.savefig('xgb_density_TestTest.png', dpi=dpi_assign, bbox_inches='tight')


##### FIGURE FOR MANUSCRIPT #####

### plot the figure -- TRAIN TEST
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, train_pred, 30, 'blue', alpha=0.1)
ax.scatter(Y_test, test_pred, 30, 'red', alpha=0.2)
ax.plot([75, 625], [75, 625], "k--", lw=2)
plt.text(0.03, 0.92, 'Train $RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, 'Test $RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.78, 'Train $R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.71, 'Test $R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured density (kg/m3)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted density (kg/m3)', labelpad=10, fontproperties=fonts)
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
plt.axis([50, 650, 50, 650])
[x.label1.set_fontfamily('arial') for x in xcoord]
[x.label1.set_fontsize(16) for x in xcoord]
[y.label1.set_fontfamily('arial') for y in ycoord]
[y.label1.set_fontsize(16) for y in ycoord]
dpi_assign = 500
plt.savefig('fig_density_TrainTest.png', dpi=dpi_assign, bbox_inches='tight')

### plot the figure -- CROSS VALIDATIONfig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, cv_pred, 30, 'purple', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.text(0.03, 0.92, '$RMSE$ = {}'.format(str(round(RMSE_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$R^2$ = {}'.format(str(round(R2_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured density (kg/m3)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted density (kg/m3)', labelpad=10, fontproperties=fonts)
import matplotlib.ticker as mticker
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
dpi_assign = 500
plt.savefig('fig_density_CrossValidation.png', dpi=dpi_assign, bbox_inches='tight')


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

# Assuming Y_train, cv_pred, RMSE_cv, R2_cv are defined earlier
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# Set spine properties
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')

# Scatter plot and line of identity
ax.scatter(Y_train, cv_pred, 30, 'purple', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)

# Add text for RMSE and R^2
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.text(0.03, 0.92, '$RMSE$ = {}'.format(str(round(RMSE_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$R^2$ = {}'.format(str(round(R2_cv, 2))), transform=ax.transAxes, fontproperties=fonts)

# Labels and font properties
plt.xlabel('Measured density (kg/m3)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted density (kg/m3)', labelpad=10, fontproperties=fonts)

# Set tickers for x and y axes
ticker_arg_x = [50, 100]  # For x-axis
ticker_arg_y = [50, 100]  # For y-axis

tickers_x = [mticker.MultipleLocator(ticker_arg_x[i]) for i in range(len(ticker_arg_x))]
tickers_y = [mticker.MultipleLocator(ticker_arg_y[i]) for i in range(len(ticker_arg_y))]

ax.xaxis.set_minor_locator(tickers_x[1])  # Apply only to x-axis
ax.yaxis.set_minor_locator(tickers_y[1])  # Apply only to y-axis

# Customize font properties of x and y-axis labels
xcoord = ax.xaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]

ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in ycoord]

# Save the figure
dpi_assign = 500
plt.savefig('fig_density_CrossValidation_v2.png', dpi=dpi_assign, bbox_inches='tight')

# Show the plot
plt.show()


##### TERMINATE #####


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import seaborn as sns

# Assuming X_train and model are defined earlier

# Create the feature importances DataFrame
feature_importances = pd.DataFrame({
    'features': X_train.columns, 
    'importance': model.feature_importances_
})

# Sort the DataFrame by 'importance' in descending order
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Keep only the top 10 most important features
feature_importances = feature_importances.head(10)

# Print the top 10 feature importances
print('\nTop 10 Feature Importances:\n')
print(feature_importances)

# Plotting the sorted feature importances
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(x=feature_importances['importance'], y=feature_importances['features'], palette='viridis')

# Customizing font properties
fonts = fm.FontProperties(family='Arial', size=16, weight='normal', style='normal')
plt.xlabel('Importance', labelpad=15, fontproperties=fonts)
plt.ylabel('Features', labelpad=15, fontproperties=fonts)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Save the plot as a PNG image
plt.savefig('xgb_top10_feature_importance.png', dpi=500, bbox_inches='tight')

plt.show()





##### PREDICTION
"""
Prediction of melting point
<Please make sure the file "candidate_desc.csv" is available before executing this section>
From here, the script excecutes prediction of melting temperatures of 110 pure compounds consisting 60 HBA and 50 HBD
<Combining the 60 HBA and 50 HBD, we will get 3000 DES combinations>
Remark: most of the pure compound structures are hypothetical!

"""
import pandas as pd

### Load descriptors for prediction
prediction = pd.read_csv("C:/Users/GKlab/Documents/PENELITIAN ZAKIAH/3rd research _ ok/density prediction/ML/XGB/descriptor_candidate_DES.csv")  # This contains 110 pure compounds
descriptors = prediction.drop('density', axis=1)  # Drop the 'density' column, which is the target variable

print('\n')
print('Descriptor data: ')
print(f'Filetype: {type(descriptors)}, Shape: {descriptors.shape}')
print(descriptors)
print(descriptors.describe())

### Predict the density of 110 pure compounds (melting point in the original code)
value_pred = model.predict(descriptors)

print('\n')
print('Prediction of descriptor data: ')
print(value_pred)
print(pd.Series(value_pred).value_counts())

# Add the predicted values as a new column called 'density'
prediction['density'] = value_pred

# Save the predictions (with the new density column) to a CSV file
value_pred_filename = "predicted_density_XGBoost_final_v1.csv"
prediction.to_csv(value_pred_filename, index=False)

print('\n')
print(f'Prediction results saved to {value_pred_filename}')


