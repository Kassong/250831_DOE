clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Novel_3_hyper.m Optimized Ensemble SVR (Hyperparameter Optimization + K-fold CV) ===\n');
fprintf('Training data: predict.csv, Test data: dataset.csv\n');
T_train = readtable('predict.csv');      % 112개
T_test  = readtable('dataset.csv');      % 16개

X_train = T_train{:,1:4}; Y_train = T_train{:,5:8};
X_test  = T_test{:,1:4};  Y_test  = T_test{:,5:8};
input_names = T_train.Properties.VariableNames(1:4);
output_names = T_train.Properties.VariableNames(5:8);
num_outputs = size(Y_train,2);

fprintf('Dataset loading completed: %d training, %d test samples\n', size(X_train,1),size(X_test,1));

% Normalization (based on training set)
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_test_norm = (X_test - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
Y_test_norm = (Y_test - Y_mean) ./ Y_std;

%% Hyperparameter optimization and cross-validation
fprintf('\n=== Hyperparameter Optimization and K-fold Cross-Validation Start ===\n');

% Hyperparameter search range settings
c_range = logspace(-2, 2, 8); % C: 0.01 ~ 100
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1
poly_orders = [2, 3]; % Polynomial orders

% K-fold settings
k_folds = 5;
n_train = size(X_train_norm, 1);

% Generate CV folds
cv_indices = zeros(n_train, 1);
indices = randperm(n_train);
fold_size = floor(n_train / k_folds);
for fold = 1:k_folds
    start_idx = (fold-1) * fold_size + 1;
    if fold == k_folds
        end_idx = n_train;
    else
        end_idx = fold * fold_size;
    end
    cv_indices(indices(start_idx:end_idx)) = fold;
end

% Result storage variables
best_params_linear = cell(num_outputs, 1);
best_params_poly = cell(num_outputs, 1);
CV_scores = zeros(num_outputs, 1); % Ensemble CV scores
CV_scores_linear = zeros(num_outputs, 1);
CV_scores_poly = zeros(num_outputs, 1);
rmse_scores = zeros(num_outputs, 1);
mae_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    fprintf('\nHyperparameter optimization for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    ytr = Y_train_norm(:,j);
    
    %% 1. Linear kernel optimization
    fprintf('  - Linear kernel hyperparameter optimization...\n');
    best_score_linear = -inf;
    best_linear = struct();
    
    for c_idx = 1:length(c_range)
        for e_idx = 1:length(epsilon_range)
            % K-fold CV
            cv_predictions = zeros(n_train, 1);
            
            for fold = 1:k_folds
                train_idx = cv_indices ~= fold;
                test_idx = cv_indices == fold;
                
                mdl = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
                    'KernelFunction', 'linear', ...
                    'BoxConstraint', c_range(c_idx), ...
                    'Epsilon', epsilon_range(e_idx), ...
                    'Standardize', false);
                
                cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
            end
            
            % R² calculation
            SS_res = sum((ytr - cv_predictions).^2);
            SS_tot = sum((ytr - mean(ytr)).^2);
            r2_score = 1 - SS_res/SS_tot;
            
            if r2_score > best_score_linear
                best_score_linear = r2_score;
                best_linear.C = c_range(c_idx);
                best_linear.epsilon = epsilon_range(e_idx);
            end
        end
    end
    
    best_params_linear{j} = best_linear;
    CV_scores_linear(j) = best_score_linear;
    fprintf('    Optimal Linear parameters: C=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_linear.C, best_linear.epsilon, best_score_linear);
    
    %% 2. Polynomial kernel optimization
    fprintf('  - Polynomial kernel hyperparameter optimization...\n');
    best_score_poly = -inf;
    best_poly = struct();
    
    for c_idx = 1:length(c_range)
        for e_idx = 1:length(epsilon_range)
            for p_idx = 1:length(poly_orders)
                % K-fold CV
                cv_predictions = zeros(n_train, 1);
                
                for fold = 1:k_folds
                    train_idx = cv_indices ~= fold;
                    test_idx = cv_indices == fold;
                    
                    mdl = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
                        'KernelFunction', 'polynomial', ...
                        'BoxConstraint', c_range(c_idx), ...
                        'PolynomialOrder', poly_orders(p_idx), ...
                        'Epsilon', epsilon_range(e_idx), ...
                        'Standardize', false);
                    
                    cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
                end
                
                % R² calculation
                SS_res = sum((ytr - cv_predictions).^2);
                SS_tot = sum((ytr - mean(ytr)).^2);
                r2_score = 1 - SS_res/SS_tot;
                
                if r2_score > best_score_poly
                    best_score_poly = r2_score;
                    best_poly.C = c_range(c_idx);
                    best_poly.epsilon = epsilon_range(e_idx);
                    best_poly.order = poly_orders(p_idx);
                end
            end
        end
    end
    
    best_params_poly{j} = best_poly;
    CV_scores_poly(j) = best_score_poly;
    fprintf('    Optimal Poly parameters: C=%.4f, epsilon=%.4f, order=%d, CV R²=%.4f\n', ...
        best_poly.C, best_poly.epsilon, best_poly.order, best_score_poly);
    
    %% 3. Ensemble performance evaluation
    fprintf('  - Ensemble model K-fold CV evaluation...\n');
    cv_predictions_ensemble = zeros(n_train, 1);
    
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        % Optimized Linear model
        mdl_lin = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', best_linear.C, ...
            'Epsilon', best_linear.epsilon, ...
            'Standardize', false);
        pred_lin = predict(mdl_lin, X_train_norm(test_idx,:));
        
        % Optimized Polynomial model
        mdl_poly = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
            'KernelFunction', 'polynomial', ...
            'BoxConstraint', best_poly.C, ...
            'PolynomialOrder', best_poly.order, ...
            'Epsilon', best_poly.epsilon, ...
            'Standardize', false);
        pred_poly = predict(mdl_poly, X_train_norm(test_idx,:));
        
        % Ensemble prediction
        cv_predictions_ensemble(test_idx) = (pred_lin + pred_poly) / 2;
    end
    
    % Ensemble R² calculation
    SS_res = sum((ytr - cv_predictions_ensemble).^2);
    SS_tot = sum((ytr - mean(ytr)).^2);
    ensemble_r2 = 1 - SS_res/SS_tot;
    CV_scores(j) = ensemble_r2;
    
    % Calculate RMSE and MAE
    rmse_scores(j) = sqrt(mean((ytr - cv_predictions_ensemble).^2));
    mae_scores(j) = mean(abs(ytr - cv_predictions_ensemble));
    
    fprintf('    Ensemble CV R²=%.4f, RMSE=%.4f, MAE=%.4f\n', ...
        ensemble_r2, rmse_scores(j), mae_scores(j));
end

fprintf('\nHyperparameter optimization completed!\n');

%% Final model training and prediction with optimized parameters
fprintf('\n=== Optimized Model Training and Prediction ===\n');

Y_pred_train_norm = zeros(size(Y_train));
Y_pred_test_norm  = zeros(size(Y_test));
RMSE_train = zeros(num_outputs,1); RMSE_test = zeros(num_outputs,1);
MAE_train  = zeros(num_outputs,1); MAE_test  = zeros(num_outputs,1);
R2_train   = zeros(num_outputs,1); R2_test   = zeros(num_outputs,1);

for j = 1:num_outputs
    fprintf('%d/%d Optimized model training and prediction: %s\n',j,num_outputs,output_names{j});
    ytr = Y_train_norm(:,j);
    
    % Train model with optimized parameters
    linear_params = best_params_linear{j};
    poly_params = best_params_poly{j};
    
    mdl_lin  = fitrsvm(X_train_norm, ytr, ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, ...
        'Standardize', false);
    
    mdl_poly = fitrsvm(X_train_norm, ytr, ...
        'KernelFunction', 'polynomial', ...
        'BoxConstraint', poly_params.C, ...
        'PolynomialOrder', poly_params.order, ...
        'Epsilon', poly_params.epsilon, ...
        'Standardize', false);
    
    % Prediction
    Y_pred_train_norm(:,j) = (predict(mdl_lin, X_train_norm) + predict(mdl_poly, X_train_norm))/2;
    Y_pred_test_norm(:,j)  = (predict(mdl_lin, X_test_norm)  + predict(mdl_poly, X_test_norm))/2;
    
    % Evaluation metrics (RMSE, MAE, R2)
    RMSE_train(j) = sqrt(mean((Y_pred_train_norm(:,j)-ytr).^2));
    RMSE_test(j)  = sqrt(mean((Y_pred_test_norm(:,j)-Y_test_norm(:,j)).^2));
    MAE_train(j)  = mean(abs(Y_pred_train_norm(:,j)-ytr));
    MAE_test(j)   = mean(abs(Y_pred_test_norm(:,j)-Y_test_norm(:,j)));
    R2_train(j)   = 1 - sum((ytr-Y_pred_train_norm(:,j)).^2)/sum((ytr-mean(ytr)).^2);
    R2_test(j)    = 1 - sum((Y_test_norm(:,j)-Y_pred_test_norm(:,j)).^2)/sum((Y_test_norm(:,j)-mean(Y_test_norm(:,j))).^2);
end
fprintf('Completed!\n');

%% Denormalization
Y_pred_train = Y_pred_train_norm .* Y_std + Y_mean;
Y_pred_test  = Y_pred_test_norm  .* Y_std + Y_mean;

%% Hold-out validation (maintained for compatibility with existing code)
fprintf('\n=== Hold-out Cross-Validation ===\n');
val_ratio = 0.2;
n_val = round(n_train * val_ratio);
val_indices = randperm(n_train);
val_idx = val_indices(1:n_val);
train_cv_idx = val_indices(n_val+1:end);

X_train_cv = X_train_norm(train_cv_idx, :);
Y_train_cv = Y_train_norm(train_cv_idx, :);
X_val_cv = X_train_norm(val_idx, :);
Y_val_cv = Y_train_norm(val_idx, :);

holdout_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    linear_params = best_params_linear{j};
    poly_params = best_params_poly{j};
    
    % Hold-out validation with optimized parameters
    mdl_lin = fitrsvm(X_train_cv, Y_train_cv(:,j), ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, ...
        'Standardize', false);
    
    mdl_poly = fitrsvm(X_train_cv, Y_train_cv(:,j), ...
        'KernelFunction', 'polynomial', ...
        'BoxConstraint', poly_params.C, ...
        'PolynomialOrder', poly_params.order, ...
        'Epsilon', poly_params.epsilon, ...
        'Standardize', false);
    
    pred_lin = predict(mdl_lin, X_val_cv);
    pred_poly = predict(mdl_poly, X_val_cv);
    val_pred = (pred_lin + pred_poly) / 2;
    
    SS_res = sum((Y_val_cv(:,j) - val_pred).^2);
    SS_tot = sum((Y_val_cv(:,j) - mean(Y_val_cv(:,j))).^2);
    holdout_scores(j) = 1 - SS_res/SS_tot;
    fprintf('  %s Hold-out R² = %.4f\n', output_names{j}, holdout_scores(j));
end

%% Complete metrics table output
fprintf('\n=== Performance Metrics Summary ===\n');
metrics_table = table(output_names', holdout_scores, CV_scores, rmse_scores, mae_scores, ...
    RMSE_train, RMSE_test, MAE_train, MAE_test, R2_train, R2_test, ...
    'VariableNames', {'Output','Holdout_R2','CV5_R2','CV_RMSE','CV_MAE', ...
    'RMSE_train','RMSE_test','MAE_train','MAE_test','R2_train','R2_test'});
disp(metrics_table);

% Optimal parameter output
fprintf('\n=== Optimal Hyperparameters ===\n');
for j = 1:num_outputs
    fprintf('%s:\n', output_names{j});
    linear_params = best_params_linear{j};
    poly_params = best_params_poly{j};
    fprintf('  Linear: C=%.4f, epsilon=%.4f (R²=%.4f)\n', ...
        linear_params.C, linear_params.epsilon, CV_scores_linear(j));
    fprintf('  Poly%d: C=%.4f, epsilon=%.4f (R²=%.4f)\n', ...
        poly_params.order, poly_params.C, poly_params.epsilon, CV_scores_poly(j));
    fprintf('  Ensemble R²=%.4f\n\n', CV_scores(j));
end

%% Visualization

% 1. Cross-validation results comparison
fprintf('\n=== Visualization Generation ===\n');
figure('Name','Optimized Cross-Validation Results','WindowStyle','docked');

subplot(2,2,1);
bar([holdout_scores, CV_scores]); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('Hold-out vs K-fold CV (Optimized)'); xtickangle(45);
legend('Hold-out', 'K-fold CV', 'Location', 'best');

subplot(2,2,2);
bar([CV_scores_linear, CV_scores_poly, CV_scores]); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('K-fold CV Performance by Component'); xtickangle(45);
legend('Linear', 'Polynomial', 'Ensemble', 'Location', 'best');

subplot(2,2,3);
bar([rmse_scores, mae_scores]); grid on;
set(gca,'XTickLabel',output_names); ylabel('Error');
title('K-fold CV Error Metrics'); xtickangle(45);
legend('RMSE', 'MAE', 'Location', 'best');

subplot(2,2,4);
x = 1:num_outputs;
width = 0.35;
bar(x - width/2, holdout_scores, width, 'FaceColor', 'b'); hold on;
bar(x + width/2, CV_scores, width, 'FaceColor', 'r');
ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('Validation Method Comparison'); xtickangle(45);
legend('Hold-out', 'K-fold CV', 'Location', 'best');

sgtitle('Hyperparameter Optimized Cross-Validation Results','FontSize',14,'FontWeight','bold');

% 2. Predicted vs actual values scatter plot
figure('Name','Optimized Ensemble SVR Prediction Correlation','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    scatter(Y_pred_test(:,j), Y_test(:,j), 120,'r','filled'); hold on;
    scatter(Y_pred_train(:,j), Y_train(:,j),90,'b','filled','MarkerFaceAlpha',0.6);
    xls = min([Y_pred_test(:,j); Y_test(:,j); Y_pred_train(:,j); Y_train(:,j)]);
    xhs = max([Y_pred_test(:,j); Y_test(:,j); Y_pred_train(:,j); Y_train(:,j)]);
    plot([xls xhs],[xls xhs],'k--','LineWidth',1.2);
    grid on; axis equal; box on; legend('Test','Training','1:1 Reference','Location','best');
    xlabel('Predicted Value'); ylabel('Actual Value');
    title(sprintf('%s (Optimized)\nRMSE_train=%.3f, test=%.3f\nMAE_train=%.3f, test=%.3f\nR²_train=%.3f, test=%.3f', ...
        output_names{j}, RMSE_train(j), RMSE_test(j), MAE_train(j), MAE_test(j), R2_train(j), R2_test(j)));
end
sgtitle('Optimized Ensemble SVR Regression Performance : Correlation Plot','FontSize',14,'FontWeight','bold');

% 3. Residual Plot
figure('Name','Optimized SVR Residual Plot','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    residual = Y_test(:,j) - Y_pred_test(:,j);
    scatter(Y_pred_test(:,j), residual, 100,'filled');
    yline(0,'k--');
    grid on; xlabel('Test Predicted Value'); ylabel('Residual (Actual-Predicted)');
    title(['Output ',output_names{j},' residual (Optimized)']);
end
sgtitle('Optimized Ensemble SVR Test Set Residual Analysis','FontSize',14);

% 4. Performance metrics bar chart
figure('Name','Optimized SVR Prediction Performance Metrics','WindowStyle','docked');
metrics = {'RMSE','MAE','R²'};
for k = 1:3
    subplot(1,3,k);
    if k==1
        bar([RMSE_train, RMSE_test],'grouped');
        set(gca,'xticklabel',output_names);
        legend('Train','Test'); ylabel('RMSE');
        title('RMSE Comparison (Optimized)');
    elseif k==2
        bar([MAE_train, MAE_test],'grouped');
        set(gca,'xticklabel',output_names);
        legend('Train','Test'); ylabel('MAE');
        title('MAE Comparison (Optimized)');
    else
        bar([R2_train, R2_test],'grouped');
        set(gca,'xticklabel',output_names);
        legend('Train','Test'); ylabel('R²');
        title('R² Comparison (Optimized)');
    end
    grid on; xtickangle(45);
end
sgtitle('Hyperparameter Optimized SVR Ensemble Performance Metrics','FontSize',14);

%% Result storage
fprintf('\n=== Results Saving ===\n');

% Save optimal hyperparameters
fprintf('Saving optimal hyperparameters to hyperparams_Novel_3.mat...\n');
save('hyperparams_Novel_3.mat', 'best_params_linear', 'best_params_poly', ...
     'CV_scores', 'CV_scores_linear', 'CV_scores_poly', ...
     'rmse_scores', 'mae_scores', 'output_names');
fprintf('Saving completed: hyperparams_Novel_3.mat\n');

% Save prediction results
fprintf('Saving optimized model prediction results to Novel_3_predictions_hyper.mat...\n');
save('Novel_3_predictions_hyper.mat', 'Y_pred_train', 'Y_pred_test', 'Y_train', 'Y_test', ...
     'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test');
fprintf('Saving completed: Novel_3_predictions_hyper.mat\n');

fprintf('\n=== Novel_3_hyper.m Execution Completed ===\n');
fprintf('Key Improvements:\n');
fprintf('1. Automatic hyperparameter optimization for Linear/Polynomial respectively\n');
fprintf('2. Improved performance evaluation with K-fold cross-validation\n');
fprintf('3. Performance analysis by components of Ensemble model\n');
fprintf('4. Automatic saving and reuse of optimal parameters\n');
fprintf('5. Enhanced visualization and detailed performance analysis\n');