# MLSSVR (Multi-output Least Squares Support Vector Regression) Project

## Overview
This project performs multi-output predictive modeling using Multi-output Least Squares Support Vector Regression with DOE (Design of Experiments) based data. MLSSVR is a specialized algorithm capable of predicting multiple output variables simultaneously, consisting of a 3-stage workflow similar to SVR.

## Dataset
- **dataset.csv**: DOE experimental data (16 conditions)
  - Input variables: X1, X2, X3, X4
  - Output variables: Y1 (Micro Ra), Y2 (Micro Rz), Y3 (Macro Ra), Y4 (Macro Rz)
  - Experimental conditions based on Taguchi orthogonal array design

## MLSSVR Library
- **MLSSVR-master/**: MLSSVR implementation library
  - `MLSSVRTrain.m`: MLSSVR model training function
  - `MLSSVRPredict.m`: MLSSVR prediction function
  - `GridMLSSVR.m`: Hyperparameter grid search
  - `Kerfun.m`: Kernel function implementation
  - `MLS-SVR.pdf`: Algorithm theory documentation

## File Structure and Execution Order

### Stage 1: Basic_1_MLSSVR.m - Basic MLSSVR Modeling
**Purpose**: Traditional MLSSVR modeling approach to establish baseline performance

**Key Features**:
- Split data into 80% training, 20% testing
- MLSSVR hyperparameter optimization (GridMLSSVR)
- Simultaneous multi-output variable prediction
- Performance metrics: R² calculation (for each output variable)
- Visualization: correlation plots, boxplots, performance comparison

**Execution**:
```matlab
run('Basic_1_MLSSVR.m')
```

**Output**:
- R² performance for each output variable
- Correlation plots and distribution comparison visualization

### Stage 2: Augment_2_MLSSVR_*.m - Prediction of Unexplored Conditions
**Purpose**: Predict remaining 112 conditions using existing 16 training data points

#### 2-1. Augment_2_MLSSVR_1G.m (Single Group approach)
- **Single Group MLSSVR**: Predict 4 output variables with one integrated model
- LOOCV (Leave-One-Out Cross-Validation) cross-validation
- Excludes 16 training conditions from total conditions (4×4×4×2 = 128)
- Generates predictions for 112 unexplored conditions

#### 2-2. Augment_2_MLSSVR_2G.m (Two Group approach)  
- **Two Group MLSSVR**: Split output variables into 2 groups
  - Group 1: Y1(Micro Ra), Y2(Micro Rz)
  - Group 2: Y3(Macro Ra), Y4(Macro Rz)
- Independent MLSSVR model training for each group
- Group-specific hyperparameter optimization

**Execution**:
```matlab
% Single group approach (recommended)
run('Augment_2_MLSSVR_1G.m')

% Two group approach
run('Augment_2_MLSSVR_2G.m')
```

**Output Files**:
- `predict_MLSSVR_1G.csv`: Single group MLSSVR prediction results
- `predict_MLSSVR_2G.csv`: Two group MLSSVR prediction results

### Stage 3: Novel_3_MLSSVR_*_hyper.m - Final Performance Validation
**Purpose**: Use Stage 2 generated predictions as training data and original dataset.csv as test set for final model performance evaluation

#### 3-1. Novel_3_MLSSVR_1G_hyper.m (Single Group approach)
- Training data: predict_MLSSVR_1G.csv (112 samples)
- Test data: dataset.csv (16 samples)
- Extended hyperparameter grid search
- Unified model predicting all output variables simultaneously

#### 3-2. Novel_3_MLSSVR_2G_hyper.m (Two Group approach)
- Training data: predict_MLSSVR_2G.csv (112 samples)
- Test data: dataset.csv (16 samples)  
- Group-wise independent model training and evaluation
- Extended hyperparameter ranges applied

**Execution**:
```matlab
% After completing Stage 2
% Single group approach (recommended)
run('Novel_3_MLSSVR_1G_hyper.m')

% Two group approach
run('Novel_3_MLSSVR_2G_hyper.m')
```

## Additional Tools

### dataAnalysis.m - Data Analysis
**Purpose**: Analyze correlations and variability of output variables

**Key Features**:
- Basic statistics of output variables (mean, standard deviation, coefficient of variation)
- Correlation coefficient matrix between output variables
- Validation of grouping rationale

**Execution**:
```matlab
run('dataAnalysis.m')
```

## Complete Workflow

```
1. Basic_1_MLSSVR.m
   ↓ (Establish performance baseline)
   
2. Augment_2_MLSSVR_*.m 
   ↓ (Generate predict_MLSSVR_*.csv files)
   
3. Novel_3_MLSSVR_*_hyper.m
   ↓ (Final performance evaluation)
   
※ Optional: dataAnalysis.m (Data characteristics analysis)
```

## MLSSVR vs SVR Differences

### MLSSVR Advantages
- **Simultaneous multi-output modeling**: Considers correlations between output variables
- **Computational efficiency**: Predicts all outputs with a single model
- **Consistent prediction**: Ensures consistency between output variables

### Hyperparameters
- **gamma (γ)**: Kernel width parameter (0.001 ~ 100)
- **lambda (λ)**: Regularization parameter (0.001 ~ 100)  
- **p**: Kernel characteristic parameter (0.1 ~ 3)

### Cross-validation Methods
- **Basic_1**: Grid Search with K-fold CV (5-fold)
- **Augment_2**: LOOCV (Leave-One-Out Cross-Validation)
- **Novel_3**: Extended Grid Search with K-fold CV

## Model Comparison Recommendations

### Single Group vs Two Groups
- **Single Group (1G)**: Recommended when output variables have high correlation
- **Two Groups (2G)**: Recommended when output variables have different characteristics

### Grouping Criteria
- Group 1: Micro surface roughness (Y1, Y2)
- Group 2: Macro surface roughness (Y3, Y4)

## Usage

### Complete Process Execution (Single Group)
```matlab
% Stage 1: Basic performance check
run('Basic_1_MLSSVR.m')

% Stage 2: Unexplored condition prediction (single group)
run('Augment_2_MLSSVR_1G.m')

% Stage 3: Final performance validation
run('Novel_3_MLSSVR_1G_hyper.m')
```

### Complete Process Execution (Two Groups)
```matlab
% Stage 1: Basic performance check
run('Basic_1_MLSSVR.m')

% Stage 2: Unexplored condition prediction (two groups)
run('Augment_2_MLSSVR_2G.m')

% Stage 3: Final performance validation
run('Novel_3_MLSSVR_2G_hyper.m')
```

### Data Analysis
```matlab
% Data characteristics analysis (optional)
run('dataAnalysis.m')
```

## Requirements
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox
- dataset.csv file must be located in the same directory
- MLSSVR-master folder must be located in the same directory

## Output Results
- Performance metrics table (console output)
- Prediction CSV files
- Correlation plots and distribution comparison visualization
- Optimal hyperparameter information
- LOOCV cross-validation results

## Notes
- All codes use `rng('default')` setting for reproducible results
- MLSSVR-master library is automatically added via `addpath` command
- Normalization is performed based on training data
- Single group approach generally provides more stable performance
- Extended hyperparameter grid search can achieve higher accuracy