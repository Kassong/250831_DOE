clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Basic_1.m SVR Model Training and Evaluation Start (Hyperparameter Optimization & K-fold CV Included) ===\n');
fprintf('Loading data...\n');
T = readtable('dataset.csv');
X = T{:,1:4}; Y = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
num_outputs = size(Y,2); num_samples = size(X,1);
fprintf('Data loading completed: %d samples, %d input variables, %d output variables\n', num_samples, length(input_names), num_outputs);

%% 1. Split (80% training, 20% testing)
fprintf('\nSplitting data...\n');
idx = randperm(num_samples);
N_train = round(0.8*num_samples);
train_idx = idx(1:N_train); test_idx = idx(N_train+1:end);

X_train = X(train_idx,:); X_test = X(test_idx,:);
Y_train = Y(train_idx,:); Y_test = Y(test_idx,:);
fprintf('Data splitting completed: %d training, %d testing\n', N_train, length(test_idx));

% Normalization (based on training set)
fprintf('Normalizing data...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_test_norm = (X_test - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
Y_test_norm = (Y_test - Y_mean) ./ Y_std;
fprintf('Data normalization completed\n');

%% Hyperparameter optimization function definition
function [best_params, best_score] = optimizeSVR(X, Y, kernel_type, cv_folds)
    % SVR hyperparameter grid search
    if strcmp(kernel_type, 'rbf')
        C_range = [0.1, 1, 10, 100];
        sigma_range = [0.1, 1, 10, 100];
        param_grid = [];
        for c = C_range
            for sigma = sigma_range
                param_grid = [param_grid; c, sigma];
            end
        end
    elseif strcmp(kernel_type, 'linear')
        C_range = [0.1, 1, 10, 100];
        param_grid = C_range';
    elseif strcmp(kernel_type, 'polynomial')
        C_range = [0.1, 1, 10, 100];
        poly_order_range = [2, 3];
        param_grid = [];
        for c = C_range
            for p = poly_order_range
                param_grid = [param_grid; c, p];
            end
        end
    end
    
    best_score = -Inf;
    best_params = param_grid(1,:);
    
    % K-fold cross-validation
    cv_partition = cvpartition(size(X,1), 'KFold', cv_folds);
    
    for i = 1:size(param_grid, 1)
        cv_scores = zeros(cv_folds, 1);
        
        for fold = 1:cv_folds
            train_idx = training(cv_partition, fold);
            val_idx = test(cv_partition, fold);
            
            X_cv_train = X(train_idx, :);
            Y_cv_train = Y(train_idx, :);
            X_cv_val = X(val_idx, :);
            Y_cv_val = Y(val_idx, :);
            
            try
                if strcmp(kernel_type, 'rbf')
                    mdl = fitrsvm(X_cv_train, Y_cv_train, 'KernelFunction', 'rbf', ...
                        'BoxConstraint', param_grid(i,1), 'KernelScale', param_grid(i,2), ...
                        'Standardize', false);
                elseif strcmp(kernel_type, 'linear')
                    mdl = fitrsvm(X_cv_train, Y_cv_train, 'KernelFunction', 'linear', ...
                        'BoxConstraint', param_grid(i,1), 'Standardize', false);
                elseif strcmp(kernel_type, 'polynomial')
                    mdl = fitrsvm(X_cv_train, Y_cv_train, 'KernelFunction', 'polynomial', ...
                        'BoxConstraint', param_grid(i,1), 'PolynomialOrder', param_grid(i,2), ...
                        'Standardize', false);
                end
                
                Y_pred_val = predict(mdl, X_cv_val);
                cv_scores(fold) = 1 - sum((Y_cv_val - Y_pred_val).^2) / sum((Y_cv_val - mean(Y_cv_val)).^2);
            catch
                cv_scores(fold) = -Inf;
            end
        end
        
        mean_cv_score = mean(cv_scores);
        if mean_cv_score > best_score
            best_score = mean_cv_score;
            best_params = param_grid(i,:);
        end
    end
end

%% 2. Kernel-based model training and prediction (including hyperparameter optimization)
fprintf('\n=== SVR Model Training Start (Hyperparameter Optimization) ===\n');
kernel_names = {'RBF','Linear','Ensemble'};
Y_pred_train_all = zeros(N_train, num_outputs, 3);
Y_pred_test_all  = zeros(num_samples-N_train, num_outputs, 3);

% Performance metrics storage variables
R2_train = zeros(num_outputs, 3);
R2_test = zeros(num_outputs, 3);
RMSE_train = zeros(num_outputs, 3);
RMSE_test = zeros(num_outputs, 3);
MAE_train = zeros(num_outputs, 3);
MAE_test = zeros(num_outputs, 3);
CV_scores = zeros(num_outputs, 3);

% K-fold settings
k_folds = 5;

for j = 1:num_outputs
    fprintf('\nTraining model for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    Yt_test = Y_test_norm(:,j);

    % RBF kernel (hyperparameter optimization)
    fprintf('  - Optimizing RBF kernel hyperparameters...\n');
    [rbf_params, rbf_cv_score] = optimizeSVR(X_train_norm, Yt_train, 'rbf', k_folds);
    fprintf('    Optimal RBF parameters: C=%.2f, Sigma=%.2f (CV R²=%.4f)\n', rbf_params(1), rbf_params(2), rbf_cv_score);
    
    mdl_rbf = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params(1), 'KernelScale', rbf_params(2), 'Standardize', false);
    Y_pred_train_all(:,j,1) = predict(mdl_rbf, X_train_norm);
    Y_pred_test_all(:,j,1) = predict(mdl_rbf, X_test_norm);
    CV_scores(j,1) = rbf_cv_score;

    % Linear kernel (hyperparameter optimization)
    fprintf('  - Optimizing Linear kernel hyperparameters...\n');
    [lin_params, lin_cv_score] = optimizeSVR(X_train_norm, Yt_train, 'linear', k_folds);
    fprintf('    Optimal Linear parameters: C=%.2f (CV R²=%.4f)\n', lin_params(1), lin_cv_score);
    
    mdl_lin = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', lin_params(1), 'Standardize', false);
    Y_pred_train_all(:,j,2) = predict(mdl_lin, X_train_norm);
    Y_pred_test_all(:,j,2) = predict(mdl_lin, X_test_norm);
    CV_scores(j,2) = lin_cv_score;

    % Ensemble (Linear + Polynomial, each optimized separately)
    fprintf('  - Optimizing Ensemble model (Linear+Poly2) hyperparameters...\n');
    [poly_params, poly_cv_score] = optimizeSVR(X_train_norm, Yt_train, 'polynomial', k_folds);
    fprintf('    Optimal Polynomial parameters: C=%.2f, Order=%d (CV R²=%.4f)\n', poly_params(1), poly_params(2), poly_cv_score);
    
    % Train ensemble model with optimized parameters
    mdl_lin_ens = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', lin_params(1), 'Standardize', false);
    mdl_poly_ens = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'polynomial', ...
        'BoxConstraint', poly_params(1), 'PolynomialOrder', poly_params(2), 'Standardize', false);
    
    Y_pred_train_lin = predict(mdl_lin_ens, X_train_norm);
    Y_pred_train_poly = predict(mdl_poly_ens, X_train_norm);
    Y_pred_test_lin = predict(mdl_lin_ens, X_test_norm);
    Y_pred_test_poly = predict(mdl_poly_ens, X_test_norm);
    
    Y_pred_train_all(:,j,3) = (Y_pred_train_lin + Y_pred_train_poly)/2;
    Y_pred_test_all(:,j,3) = (Y_pred_test_lin + Y_pred_test_poly)/2;
    CV_scores(j,3) = (lin_cv_score + poly_cv_score)/2; % Ensemble CV score
    
    % Calculate performance metrics (for all kernels)
    for k = 1:3
        % R² calculation
        R2_train(j,k) = 1 - sum((Yt_train - Y_pred_train_all(:,j,k)).^2) / sum((Yt_train - mean(Yt_train)).^2);
        R2_test(j,k) = 1 - sum((Yt_test - Y_pred_test_all(:,j,k)).^2) / sum((Yt_test - mean(Yt_test)).^2);
        
        % RMSE calculation
        RMSE_train(j,k) = sqrt(mean((Yt_train - Y_pred_train_all(:,j,k)).^2));
        RMSE_test(j,k) = sqrt(mean((Yt_test - Y_pred_test_all(:,j,k)).^2));
        
        % MAE calculation
        MAE_train(j,k) = mean(abs(Yt_train - Y_pred_train_all(:,j,k)));
        MAE_test(j,k) = mean(abs(Yt_test - Y_pred_test_all(:,j,k)));
    end
    
    fprintf('  %s completed:\n', output_names{j});
    fprintf('    RBF: R²=%.3f, RMSE=%.3f, MAE=%.3f\n', R2_test(j,1), RMSE_test(j,1), MAE_test(j,1));
    fprintf('    Linear: R²=%.3f, RMSE=%.3f, MAE=%.3f\n', R2_test(j,2), RMSE_test(j,2), MAE_test(j,2));
    fprintf('    Ensemble: R²=%.3f, RMSE=%.3f, MAE=%.3f\n', R2_test(j,3), RMSE_test(j,3), MAE_test(j,3));
end
fprintf('\nAll model training completed!\n');

%% 3. Denormalization
fprintf('\nDenormalizing predictions...\n');
Y_pred_train_denorm = zeros(size(Y_pred_train_all));
Y_pred_test_denorm = zeros(size(Y_pred_test_all));

for k = 1:3
    for j = 1:num_outputs
        Y_pred_train_denorm(:,j,k) = Y_pred_train_all(:,j,k) * Y_std(j) + Y_mean(j);
        Y_pred_test_denorm(:,j,k)  = Y_pred_test_all(:,j,k)  * Y_std(j) + Y_mean(j);
    end
end
fprintf('Denormalization completed\n');

%% 4. Select optimal kernel based on test performance for each output and performance metrics table
fprintf('\n=== Optimal Kernel Selection ===\n');
[~,best_idx] = max(R2_test,[],2);
Yfit_train_best = zeros(N_train, num_outputs);
Yfit_test_best  = zeros(num_samples-N_train, num_outputs);
kernel_best = cell(num_outputs,1);

for j = 1:num_outputs
    Yfit_train_best(:,j) = Y_pred_train_denorm(:,j,best_idx(j));
    Yfit_test_best(:,j)  = Y_pred_test_denorm(:,j,best_idx(j));
    kernel_best{j} = kernel_names{best_idx(j)};
    fprintf('%s: %s kernel selected (R²=%.3f, RMSE=%.3f, MAE=%.3f, CV=%.3f)\n', ...
        output_names{j}, kernel_best{j}, R2_test(j,best_idx(j)), ...
        RMSE_test(j,best_idx(j)), MAE_test(j,best_idx(j)), CV_scores(j,best_idx(j)));
end

% Generate comprehensive performance metrics table
fprintf('\n=== Comprehensive Performance Metrics Table ===\n');
fprintf('%-10s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n', ...
    'Output', 'Kernel', 'CV_R2', 'R2_train', 'R2_test', 'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test');
fprintf('==========================================================================================\n');

for j = 1:num_outputs
    best_k = best_idx(j);
    fprintf('%-10s %-10s %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n', ...
        output_names{j}, kernel_best{j}, CV_scores(j,best_k), ...
        R2_train(j,best_k), R2_test(j,best_k), RMSE_train(j,best_k), RMSE_test(j,best_k), ...
        MAE_train(j,best_k), MAE_test(j,best_k));
end

%% 5. K-fold CV results visualization
fprintf('\nGenerating visualizations...\n');
figure('Name','K-fold Cross-Validation Results','WindowStyle','docked');
for j = 1:num_outputs
    subplot(1,num_outputs,j);
    bar(CV_scores(j,:)); ylim([0 1]); grid on;
    set(gca,'XTickLabel',kernel_names); ylabel('CV R²');
    title(['Output ',output_names{j}]);
    % Mark best performance
    [max_val, max_idx] = max(CV_scores(j,:));
    text(max_idx, max_val+0.02, sprintf('%.3f', max_val), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'red');
end
sgtitle('SVR K-fold Cross-Validation R² by Kernel','FontSize',14,'FontWeight','bold');

%% 6. Performance metrics comparison (RMSE, MAE, R²)
figure('Name','Performance Metrics Comparison','WindowStyle','docked');

% 6-1. R² 비교
subplot(2,2,1);
bar(R2_test'); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test R²');
title('Test R² Comparison'); legend(kernel_names, 'Location', 'best');
xtickangle(45);

% 6-2. RMSE 비교
subplot(2,2,2);
bar(RMSE_test'); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test RMSE');
title('Test RMSE Comparison'); legend(kernel_names, 'Location', 'best');
xtickangle(45);

% 6-3. MAE 비교
subplot(2,2,3);
bar(MAE_test'); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test MAE');
title('Test MAE Comparison'); legend(kernel_names, 'Location', 'best');
xtickangle(45);

% 6-4. CV vs Test R² comparison
subplot(2,2,4);
hold on;
for k = 1:3
    scatter(CV_scores(:,k), R2_test(:,k), 100, 'filled', 'DisplayName', kernel_names{k});
end
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
xlabel('CV R²'); ylabel('Test R²'); 
title('CV vs Test R² Correlation'); legend('Location', 'best');
grid on; axis equal; xlim([0 1]); ylim([0 1]);

sgtitle('SVR Performance Metrics Comparison','FontSize',14,'FontWeight','bold');

%% 7. Predicted vs Actual CORRELATION PLOT (all outputs, all kernels) - training + test sets
% Color settings by kernel
kernel_colors = {'r', 'g', 'b'}; % RBF=red, Linear=green, Ensemble=blue
kernel_markers = {'o', 's', '^'}; % RBF=circle, Linear=square, Ensemble=triangle

figure('Name','Correlation Plots: All Kernel Performance (Y1-Y4, Training+Test)','WindowStyle','docked');

for j = 1:num_outputs
    % One subplot per output variable
    subplot(2,2,j);
    
    % Plot training set results for all kernels (with transparency)
    hold on;
    for k = 1:3
        % Training set - small and transparent
        scatter(Y_pred_train_denorm(:,j,k), Y_train(:,j), 60, ...
            kernel_colors{k}, kernel_markers{k}, 'MarkerFaceAlpha', 0.4, ...
            'MarkerEdgeAlpha', 0.6, ...
            'DisplayName', sprintf('%s Train (R²=%.3f)', kernel_names{k}, R2_train(j,k)));
    end
    
    % Plot test set results for all kernels (bold)
    for k = 1:3
        % Test set - large and bold
        scatter(Y_pred_test_denorm(:,j,k), Y_test(:,j), 120, ...
            kernel_colors{k}, 'filled', kernel_markers{k}, ...
            'DisplayName', sprintf('%s Test (R²=%.3f)', kernel_names{k}, R2_test(j,k)));
    end
    
    % 1:1 reference line
    y_all = [Y_train(:,j); Y_test(:,j)];
    x_all = [Y_pred_train_denorm(:,j,:); Y_pred_test_denorm(:,j,:)];
    ymin = min(y_all); ymax = max(y_all);
    xmin = min(x_all(:)); xmax = max(x_all(:));
    lim_min = min(ymin, xmin); lim_max = max(ymax, xmax);
    plot([lim_min lim_max], [lim_min lim_max], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
    
    grid on; axis equal; box on;
    xlabel('Predicted'); ylabel('Actual');
    title(sprintf('%s\nTraining+Test All Kernel Performance', output_names{j}));
    legend('Location', 'best', 'FontSize', 6);
    
    % Set axis range
    xlim([lim_min*0.95 lim_max*1.05]);
    ylim([lim_min*0.95 lim_max*1.05]);
end

sgtitle('SVR All Kernel Performance Comparison (Y1-Y4 Training+Test Results)','FontSize',14,'FontWeight','bold');

%% 7-1. Training set only CORRELATION PLOT
figure('Name','Correlation Plots: Training Set Only (Y1-Y4, All Kernels)','WindowStyle','docked');

for j = 1:num_outputs
    % One subplot per output variable
    subplot(2,2,j);
    
    % Plot only training set results for all kernels
    hold on;
    for k = 1:3
        scatter(Y_pred_train_denorm(:,j,k), Y_train(:,j), 100, ...
            kernel_colors{k}, 'filled', kernel_markers{k}, ...
            'DisplayName', sprintf('%s (R²=%.3f)', kernel_names{k}, R2_train(j,k)));
    end
    
    % 1:1 reference line
    ymin = min(Y_train(:,j)); ymax = max(Y_train(:,j));
    xmin = min(Y_pred_train_denorm(:,j,:),[], 'all'); 
    xmax = max(Y_pred_train_denorm(:,j,:),[], 'all');
    lim_min = min(ymin, xmin); lim_max = max(ymax, xmax);
    plot([lim_min lim_max], [lim_min lim_max], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
    
    grid on; axis equal; box on;
    xlabel('Predicted'); ylabel('Actual');
    title(sprintf('%s\nTraining Set All Kernel Performance', output_names{j}));
    legend('Location', 'best', 'FontSize', 8);
    
    % Set axis range
    xlim([lim_min*0.95 lim_max*1.05]);
    ylim([lim_min*0.95 lim_max*1.05]);
end

sgtitle('SVR All Kernel Performance Comparison (Y1-Y4 Training Set Results)','FontSize',14,'FontWeight','bold');

%% 7-2. Test set only CORRELATION PLOT
figure('Name','Correlation Plots: Test Set Only (Y1-Y4, All Kernels)','WindowStyle','docked');

for j = 1:num_outputs
    % One subplot per output variable
    subplot(2,2,j);
    
    % Plot only test set results for all kernels
    hold on;
    for k = 1:3
        scatter(Y_pred_test_denorm(:,j,k), Y_test(:,j), 100, ...
            kernel_colors{k}, 'filled', kernel_markers{k}, ...
            'DisplayName', sprintf('%s (R²=%.3f)', kernel_names{k}, R2_test(j,k)));
    end
    
    % 1:1 reference line
    ymin = min(Y_test(:,j)); ymax = max(Y_test(:,j));
    xmin = min(Y_pred_test_denorm(:,j,:),[], 'all'); 
    xmax = max(Y_pred_test_denorm(:,j,:),[], 'all');
    lim_min = min(ymin, xmin); lim_max = max(ymax, xmax);
    plot([lim_min lim_max], [lim_min lim_max], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
    
    grid on; axis equal; box on;
    xlabel('Predicted'); ylabel('Actual');
    title(sprintf('%s\nTest Set All Kernel Performance', output_names{j}));
    legend('Location', 'best', 'FontSize', 8);
    
    % Set axis range
    xlim([lim_min*0.95 lim_max*1.05]);
    ylim([lim_min*0.95 lim_max*1.05]);
end

sgtitle('SVR All Kernel Performance Comparison (Y1-Y4 Test Set Results)','FontSize',14,'FontWeight','bold');

%% 8. Detailed performance analysis - Bar chart of all kernel performance by output variable
figure('Name','Detailed Performance Analysis: All Kernels by Output Variable','WindowStyle','docked');

% R² performance comparison
subplot(2,2,1);
bar(R2_test'); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test R²');
title('Test R² Comparison (All Kernels)'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% Display values on top of each bar
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, R2_test(j,k) + 0.02, sprintf('%.3f', R2_test(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% RMSE performance comparison
subplot(2,2,2);
bar(RMSE_test'); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test RMSE');
title('Test RMSE Comparison (All Kernels)'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% Display values on top of each bar
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, RMSE_test(j,k) + max(RMSE_test(:,k))*0.02, sprintf('%.3f', RMSE_test(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% MAE performance comparison
subplot(2,2,3);
bar(MAE_test'); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test MAE');
title('Test MAE Comparison (All Kernels)'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% Display values on top of each bar
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, MAE_test(j,k) + max(MAE_test(:,k))*0.02, sprintf('%.3f', MAE_test(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% CV score comparison
subplot(2,2,4);
bar(CV_scores'); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('CV R²');
title('Cross-Validation R² Comparison'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% Display values on top of each bar
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, CV_scores(j,k) + 0.02, sprintf('%.3f', CV_scores(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

sgtitle('Detailed Performance Analysis: Y1-Y4 All Kernel Performance Metrics','FontSize',14,'FontWeight','bold');

%% 9. Performance matrix visualization using heatmaps
figure('Name','Performance Heatmap: Output Variables×Kernels','WindowStyle','docked');

% R² heatmap
subplot(2,2,1);
imagesc(R2_test); colorbar; colormap(gca, 'hot');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('Test R² Heatmap'); xlabel('Kernel'); ylabel('Output Variables');
% Display values
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', R2_test(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% RMSE heatmap
subplot(2,2,2);
imagesc(RMSE_test); colorbar; colormap(gca, 'cool');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('Test RMSE Heatmap'); xlabel('Kernel'); ylabel('Output Variables');
% Display values
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', RMSE_test(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% MAE heatmap
subplot(2,2,3);
imagesc(MAE_test); colorbar; colormap(gca, 'winter');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('Test MAE Heatmap'); xlabel('Kernel'); ylabel('Output Variables');
% Display values
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', MAE_test(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% CV score heatmap
subplot(2,2,4);
imagesc(CV_scores); colorbar; colormap(gca, 'spring');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('CV R² Heatmap'); xlabel('Kernel'); ylabel('Output Variables');
% Display values
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', CV_scores(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'black', 'FontWeight', 'bold');
    end
end

sgtitle('Performance Heatmap: Output Variables×Kernel Matrix','FontSize',14,'FontWeight','bold');

%% 8. Box plot distribution comparison (training/test/prediction, by kernel)
figure('Name','Prediction Distribution Boxplot (Each Kernel)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Yfit_train_best(:,j); Y_test(:,j); Yfit_test_best(:,j)];
    labels_box = [repmat({'Actual Training'},N_train,1); repmat({'Predicted Training'},N_train,1); ...
                  repmat({'Actual Test'},num_samples-N_train,1); repmat({'Predicted Test'},num_samples-N_train,1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('SVR Distribution Comparison (Training/Test/Prediction)','FontSize',14,'FontWeight','bold');
