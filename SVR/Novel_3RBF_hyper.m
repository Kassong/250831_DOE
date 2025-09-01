clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Novel_3RBF_hyper.m Optimized RBF SVR (Hyperparameter Optimization + K-fold CV) ===\n');
fprintf('Training data: predict_rbf.csv, Test data: dataset.csv\n');
T_train = readtable('predict_rbf.csv');      % 112개
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
gamma_range = logspace(-4, 1, 8); % gamma: 0.0001 ~ 10 (for RBF)
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1

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
best_params = cell(num_outputs, 1);
CV_scores = zeros(num_outputs, 1);
rmse_scores = zeros(num_outputs, 1);
mae_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    fprintf('\nRBF SVR hyperparameter optimization for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    ytr = Y_train_norm(:,j);
    
    % RBF kernel optimization
    fprintf('  - RBF kernel hyperparameter optimization...\n');
    best_score = -inf;
    best_rbf = struct();
    
    for c_idx = 1:length(c_range)
        for g_idx = 1:length(gamma_range)
            for e_idx = 1:length(epsilon_range)
                % K-fold CV
                cv_predictions = zeros(n_train, 1);
                
                for fold = 1:k_folds
                    train_idx = cv_indices ~= fold;
                    test_idx = cv_indices == fold;
                    
                    mdl = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
                        'KernelFunction', 'rbf', ...
                        'BoxConstraint', c_range(c_idx), ...
                        'KernelScale', 1/sqrt(2*gamma_range(g_idx)), ...
                        'Epsilon', epsilon_range(e_idx), ...
                        'Standardize', false);
                    
                    cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
                end
                
                % R² calculation
                SS_res = sum((ytr - cv_predictions).^2);
                SS_tot = sum((ytr - mean(ytr)).^2);
                r2_score = 1 - SS_res/SS_tot;
                
                if r2_score > best_score
                    best_score = r2_score;
                    best_rbf.C = c_range(c_idx);
                    best_rbf.gamma = gamma_range(g_idx);
                    best_rbf.epsilon = epsilon_range(e_idx);
                end
            end
        end
    end
    
    best_params{j} = best_rbf;
    CV_scores(j) = best_score;
    
    fprintf('    Optimal RBF parameters: C=%.4f, gamma=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_rbf.C, best_rbf.gamma, best_rbf.epsilon, best_score);
    
    % Calculate RMSE and MAE
    cv_predictions_final = zeros(n_train, 1);
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        mdl = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', best_rbf.C, ...
            'KernelScale', 1/sqrt(2*best_rbf.gamma), ...
            'Epsilon', best_rbf.epsilon, ...
            'Standardize', false);
        
        cv_predictions_final(test_idx) = predict(mdl, X_train_norm(test_idx,:));
    end
    
    rmse_scores(j) = sqrt(mean((ytr - cv_predictions_final).^2));
    mae_scores(j) = mean(abs(ytr - cv_predictions_final));
    
    fprintf('    CV RMSE=%.4f, MAE=%.4f\n', rmse_scores(j), mae_scores(j));
end

fprintf('\nHyperparameter optimization completed!\n');

%% Final model training and prediction with optimized parameters
fprintf('\n=== Optimized RBF SVR Model Training and Prediction ===\n');

Y_pred_train_norm = zeros(size(Y_train));
Y_pred_test_norm  = zeros(size(Y_test));
RMSE_train = zeros(num_outputs,1); RMSE_test = zeros(num_outputs,1);
MAE_train  = zeros(num_outputs,1); MAE_test  = zeros(num_outputs,1);
R2_train   = zeros(num_outputs,1); R2_test   = zeros(num_outputs,1);

for j = 1:num_outputs
    fprintf('%d/%d Optimized RBF SVR training and prediction: %s\n',j,num_outputs,output_names{j});
    ytr = Y_train_norm(:,j);
    
    % Train model with optimized parameters
    rbf_params = best_params{j};
    
    mdl_rbf = fitrsvm(X_train_norm, ytr, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params.C, ...
        'KernelScale', 1/sqrt(2*rbf_params.gamma), ...
        'Epsilon', rbf_params.epsilon, ...
        'Standardize', false);
    
    % Prediction
    Y_pred_train_norm(:,j) = predict(mdl_rbf, X_train_norm);
    Y_pred_test_norm(:,j)  = predict(mdl_rbf, X_test_norm);
    
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
    rbf_params = best_params{j};
    
    % Hold-out validation with optimized parameters
    mdl_rbf = fitrsvm(X_train_cv, Y_train_cv(:,j), ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params.C, ...
        'KernelScale', 1/sqrt(2*rbf_params.gamma), ...
        'Epsilon', rbf_params.epsilon, ...
        'Standardize', false);
    
    val_pred = predict(mdl_rbf, X_val_cv);
    
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
    rbf_params = best_params{j};
    fprintf('%s: C=%.4f, gamma=%.4f, epsilon=%.4f (CV R²=%.4f)\n', ...
        output_names{j}, rbf_params.C, rbf_params.gamma, rbf_params.epsilon, CV_scores(j));
end

%% Visualization

% 1. Cross-validation results comparison
fprintf('\n=== Visualization Generation ===\n');
figure('Name','Optimized RBF SVR Cross-Validation Results','WindowStyle','docked');

subplot(2,2,1);
bar([holdout_scores, CV_scores]); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('Hold-out vs K-fold CV (Optimized)'); xtickangle(45);
legend('Hold-out', 'K-fold CV', 'Location', 'best');

subplot(2,2,2);
bar([CV_scores, rmse_scores, mae_scores]); grid on;
set(gca,'XTickLabel',output_names); ylabel('Performance Metrics');
title('K-fold CV Performance Metrics'); xtickangle(45);
legend('R²', 'RMSE', 'MAE', 'Location', 'best');

subplot(2,2,3);
x = 1:num_outputs;
width = 0.35;
bar(x - width/2, holdout_scores, width, 'FaceColor', 'b'); hold on;
bar(x + width/2, CV_scores, width, 'FaceColor', 'r');
ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('Validation Method Comparison'); xtickangle(45);
legend('Hold-out', 'K-fold CV', 'Location', 'best');

subplot(2,2,4);
% Optimal parameter gamma value visualization
gamma_values = zeros(num_outputs, 1);
for j = 1:num_outputs
    gamma_values(j) = best_params{j}.gamma;
end
semilogy(1:num_outputs, gamma_values, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
set(gca,'XTickLabel',output_names); ylabel('Optimal Gamma Value (log scale)');
title('Optimal Gamma Parameter by Output'); xtickangle(45); grid on;

sgtitle('Hyperparameter Optimized RBF SVR Cross-Validation Results','FontSize',14,'FontWeight','bold');

% 2. Predicted vs actual values scatter plot
figure('Name','Optimized RBF SVR Prediction Correlation','WindowStyle','docked');
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
sgtitle('Optimized RBF SVR Regression Performance : Correlation Plot','FontSize',14,'FontWeight','bold');

% 3. Residual Plot
figure('Name','Optimized RBF SVR Residual Plot','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    residual = Y_test(:,j) - Y_pred_test(:,j);
    scatter(Y_pred_test(:,j), residual, 100,'filled','MarkerFaceColor','b');
    yline(0,'k--');
    grid on; xlabel('Test Predicted Value'); ylabel('Residual (Actual-Predicted)');
    title(['Output ',output_names{j},' residual (Optimized)']);
end
sgtitle('Optimized RBF SVR Test Set Residual Analysis','FontSize',14);

% 4. Performance metrics bar chart
figure('Name','Optimized RBF SVR Prediction Performance Metrics','WindowStyle','docked');
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
sgtitle('Hyperparameter Optimized RBF SVR Performance Metrics','FontSize',14);

% 5. Hyperparameter visualization
figure('Name','Optimal RBF Hyperparameter Analysis','WindowStyle','docked');

% C value distribution
subplot(2,2,1);
C_values = zeros(num_outputs, 1);
for j = 1:num_outputs
    C_values(j) = best_params{j}.C;
end
semilogy(1:num_outputs, C_values, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'r');
set(gca,'XTickLabel',output_names); ylabel('Optimal C Value (log scale)');
title('Optimal C Parameter by Output'); xtickangle(45); grid on;

% gamma value distribution
subplot(2,2,2);
semilogy(1:num_outputs, gamma_values, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'g');
set(gca,'XTickLabel',output_names); ylabel('Optimal Gamma Value (log scale)');
title('Optimal Gamma Parameter by Output'); xtickangle(45); grid on;

% epsilon value distribution
subplot(2,2,3);
epsilon_values = zeros(num_outputs, 1);
for j = 1:num_outputs
    epsilon_values(j) = best_params{j}.epsilon;
end
semilogy(1:num_outputs, epsilon_values, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'b');
set(gca,'XTickLabel',output_names); ylabel('Optimal Epsilon Value (log scale)');
title('Optimal Epsilon Parameter by Output'); xtickangle(45); grid on;

% Parameter correlation
subplot(2,2,4);
scatter(log10(C_values), log10(gamma_values), 100, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel('log₁₀(C)'); ylabel('log₁₀(gamma)');
title('C vs Gamma Parameter Relationship'); grid on;
for j = 1:num_outputs
    text(log10(C_values(j)), log10(gamma_values(j)), output_names{j}, ...
        'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

sgtitle('Optimal RBF Hyperparameter Analysis','FontSize',14,'FontWeight','bold');

%% Result storage
fprintf('\n=== Results Saving ===\n');

% Save optimal hyperparameters
fprintf('Saving optimal hyperparameters to hyperparams_Novel_3RBF.mat...\n');
save('hyperparams_Novel_3RBF.mat', 'best_params', 'CV_scores', ...
     'rmse_scores', 'mae_scores', 'output_names');
fprintf('Saving completed: hyperparams_Novel_3RBF.mat\n');

% Save prediction results
fprintf('Saving optimized RBF SVR prediction results to Novel_3RBF_predictions_hyper.mat...\n');
save('Novel_3RBF_predictions_hyper.mat', 'Y_pred_train', 'Y_pred_test', 'Y_train', 'Y_test', ...
     'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test');
fprintf('Saving completed: Novel_3RBF_predictions_hyper.mat\n');

fprintf('\n=== Novel_3RBF_hyper.m Execution Completed ===\n');
fprintf('Key Improvements:\n');
fprintf('1. RBF SVR automatic hyperparameter optimization (Grid Search: C, gamma, epsilon)\n');
fprintf('2. Improved performance evaluation with K-fold cross-validation\n');
fprintf('3. Automatic detection and application of optimal C, gamma, epsilon parameters\n');
fprintf('4. Automatic saving and reuse of optimal parameters\n');
fprintf('5. Added hyperparameter analysis visualization\n');
fprintf('6. Enhanced performance analysis and detailed result reporting\n');