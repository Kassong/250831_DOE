# SVR (Support Vector Regression) Project

## Overview
This project performs predictive modeling using Support Vector Regression with DOE (Design of Experiments) based data. It consists of a 3-stage workflow, each with different purposes and methodologies.

## Dataset
- **dataset.csv**: DOE experimental data (16 conditions)
  - Input variables: X1, X2, X3, X4
  - Output variables: Y1, Y2, Y3, Y4
  - Experimental conditions based on Taguchi orthogonal array design

## File Structure and Execution Order

### Stage 1: Basic_1.m - Basic SVR Modeling
**Purpose**: Traditional SVR modeling approach to establish baseline performance

**Key Features**:
- Split data into 80% training, 20% testing
- Compare 3 kernels (RBF, Linear, Ensemble)
- Hyperparameter optimization through K-fold cross-validation
- Performance metrics: R², RMSE, MAE calculation
- Comprehensive visualization (correlation plots, performance comparison, heatmaps, etc.)

**Execution**:
```matlab
run('Basic_1.m')
```

**Output**:
- Optimal kernel selection for each output variable
- Comprehensive performance metrics table
- Various visualization graphs

### Stage 2: Augment_2_*.m - Prediction of Unexplored Conditions
**Purpose**: Predict remaining 112 conditions using existing 16 training data points

#### 2-1. Augment_2_hyper.m (Ensemble approach)
- Uses RBF, Linear, and Ensemble kernels
- Excludes 16 training conditions from total conditions (4×4×4×2 = 128)
- Generates predictions for 112 unexplored conditions

#### 2-2. Augment_2linear_hyper.m (Linear kernel)
- Uses only Linear kernel with optimization
- Generates predict_linear.csv

#### 2-3. Augment_2rbf_hyper.m (RBF kernel)  
- Uses only RBF kernel with optimization
- Generates predict_rbf.csv

**Execution**:
```matlab
% Ensemble approach (recommended)
run('Augment_2_hyper.m')

% Individual kernel approaches
run('Augment_2linear_hyper.m')
run('Augment_2rbf_hyper.m')
```

**Output Files**:
- `predict.csv`: Ensemble approach prediction results
- `predict_linear.csv`: Linear kernel prediction results  
- `predict_rbf.csv`: RBF kernel prediction results

### Stage 3: Novel_3_*.m - Final Performance Validation
**Purpose**: Use Stage 2 generated predictions as training data and original dataset.csv as test set for final model performance evaluation

#### 3-1. Novel_3_hyper.m (Ensemble approach)
- Training data: predict.csv (112 samples)
- Test data: dataset.csv (16 samples)
- Linear + Polynomial ensemble model

#### 3-2. Novel_3Linear_hyper.m (Linear kernel)
- Training data: predict_linear.csv (112 samples)
- Test data: dataset.csv (16 samples)

#### 3-3. Novel_3RBF_hyper.m (RBF kernel)
- Training data: predict_rbf.csv (112 samples)  
- Test data: dataset.csv (16 samples)

**Execution**:
```matlab
% After completing Stage 2
% Ensemble approach (recommended)
run('Novel_3_hyper.m')

% Individual kernel approaches
run('Novel_3Linear_hyper.m')
run('Novel_3RBF_hyper.m')
```

## Complete Workflow

```
1. Basic_1.m
   ↓ (Establish performance baseline)
   
2. Augment_2_*.m 
   ↓ (Generate predict*.csv files)
   
3. Novel_3_*.m
   ↓ (Final performance evaluation)
```

## Key Features

### Hyperparameter Optimization
- **Grid Search**: Systematic parameter exploration
- **K-fold Cross-validation**: Prevent overfitting and improve generalization
- **Parameter Ranges**:
  - C (Regularization): 0.01 ~ 100
  - Gamma (RBF): 0.0001 ~ 10
  - Epsilon: 0.001 ~ 0.1

### Performance Metrics
- **R² (Coefficient of Determination)**: Model explanatory power
- **RMSE (Root Mean Square Error)**: Prediction error
- **MAE (Mean Absolute Error)**: Absolute error

### Visualization
- Cross-validation results comparison
- Predicted vs Actual correlation plots
- Kernel performance comparison charts
- Performance matrix visualization via heatmaps

## Usage

### Complete Process Execution
```matlab
% Stage 1: Basic performance check
run('Basic_1.m')

% Stage 2: Unexplored condition prediction (ensemble approach recommended)
run('Augment_2_hyper.m')

% Stage 3: Final performance validation
run('Novel_3_hyper.m')
```

### Individual Kernel Execution
```matlab
% Linear kernel only
run('Augment_2linear_hyper.m')
run('Novel_3Linear_hyper.m')

% RBF kernel only  
run('Augment_2rbf_hyper.m')
run('Novel_3RBF_hyper.m')
```

## Requirements
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox
- dataset.csv file must be located in the same directory

## Output Results
- Performance metrics table (console output)
- Prediction CSV files
- Various visualization graphs
- Optimal hyperparameter information

## Notes
- All codes use `rng('default')` setting for reproducible results
- Normalization is performed based on training data

