clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Augment_2rbf_hyper.m RBF SVR Untested Condition Prediction (Hyperparameter Optimization + K-fold CV) ===\n');
fprintf('Loading data...\n');
T = readtable('dataset.csv');
X_train = T{:,1:4}; Y_train = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
num_outputs = size(Y_train,2);
fprintf('Training data: %d samples, %d output variables\n', size(X_train,1), num_outputs);

%% 1. Generate complete conditions for Taguchi OA
fprintf('\nGenerating Taguchi orthogonal array complete conditions...\n');
x1_values = [250, 750, 1250, 1750];
x2_values = [20, 40, 60, 80];
x3_values = [150, 300, 450, 600];
x4_values = [4, 8];
[X1, X2, X3, X4] = ndgrid(x1_values, x2_values, x3_values, x4_values);
X_all = [X1(:), X2(:), X3(:), X4(:)];
fprintf('Total conditions: %d (%d×%d×%d×%d)\n', size(X_all,1), length(x1_values), length(x2_values), length(x3_values), length(x4_values));

% Extract 112 prediction target conditions excluding training set
is_train = ismember(X_all, X_train, 'rows');
X_predict = X_all(~is_train,:);
fprintf('Prediction targets after excluding training conditions: %d conditions\n', size(X_predict,1));

%% 2. Input and output normalization (based on training set)
fprintf('\nNormalizing data...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_predict_norm = (X_predict - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
fprintf('Normalization completed (based on training set)\n');

%% 3. RBF SVR hyperparameter optimization and K-fold cross-validation
fprintf('\n=== RBF SVR Hyperparameter Optimization and K-fold CV Start ===\n');

% Hyperparameter search range settings
c_range = logspace(-2, 2, 8); % C: 0.01 ~ 100
gamma_range = logspace(-4, 1, 8); % gamma: 0.0001 ~ 10 (RBF용)
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1

% K-fold settings
k_folds = 5; % 5-fold CV (16개 샘플이므로)
n_samples = size(X_train_norm, 1);

% Generate CV partitions
cv_indices = zeros(n_samples, 1);
indices = randperm(n_samples);
fold_size = floor(n_samples / k_folds);
for fold = 1:k_folds
    start_idx = (fold-1) * fold_size + 1;
    if fold == k_folds
        end_idx = n_samples;
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
    fprintf('\nOptimizing RBF SVR for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % RBF kernel optimization
    fprintf('  - RBF kernel hyperparameter optimization...\n');
    best_score = -inf;
    best_rbf_params = struct();
    
    for c_idx = 1:length(c_range)
        for g_idx = 1:length(gamma_range)
            for e_idx = 1:length(epsilon_range)
                % K-fold CV
                cv_predictions = zeros(n_samples, 1);
                
                for fold = 1:k_folds
                    train_idx = cv_indices ~= fold;
                    test_idx = cv_indices == fold;
                    
                    mdl = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
                        'KernelFunction', 'rbf', 'BoxConstraint', c_range(c_idx), ...
                        'KernelScale', 1/sqrt(2*gamma_range(g_idx)), ...
                        'Epsilon', epsilon_range(e_idx), 'Standardize', false);
                    
                    cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
                end
                
                % R² calculation
                SS_res = sum((Yt_train - cv_predictions).^2);
                SS_tot = sum((Yt_train - mean(Yt_train)).^2);
                r2_score = 1 - SS_res/SS_tot;
                
                if r2_score > best_score
                    best_score = r2_score;
                    best_rbf_params.C = c_range(c_idx);
                    best_rbf_params.gamma = gamma_range(g_idx);
                    best_rbf_params.epsilon = epsilon_range(e_idx);
                end
            end
        end
    end
    
    best_params{j} = best_rbf_params;
    CV_scores(j) = best_score;
    fprintf('    Optimal RBF parameters: C=%.4f, gamma=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_rbf_params.C, best_rbf_params.gamma, best_rbf_params.epsilon, best_score);
    
    % Calculate RMSE and MAE (with optimal parameters)
    cv_pred = zeros(n_samples, 1);
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        mdl = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
            'KernelFunction', 'rbf', 'BoxConstraint', best_rbf_params.C, ...
            'KernelScale', 1/sqrt(2*best_rbf_params.gamma), ...
            'Epsilon', best_rbf_params.epsilon, 'Standardize', false);
        cv_pred(test_idx) = predict(mdl, X_train_norm(test_idx,:));
    end
    
    rmse_scores(j) = sqrt(mean((Yt_train - cv_pred).^2));
    mae_scores(j) = mean(abs(Yt_train - cv_pred));
    
    fprintf('    CV RMSE=%.4f, MAE=%.4f\n', rmse_scores(j), mae_scores(j));
end

fprintf('\nRBF SVR hyperparameter optimization completed!\n');

%% 4. Optimized RBF SVR model training and prediction
fprintf('\n=== Optimized RBF SVR Model Training and Prediction Start ===\n');
Y_predict_norm = zeros(size(X_predict,1), num_outputs);

for j = 1:num_outputs
    fprintf('Predicting optimized RBF for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % Optimized RBF SVR
    fprintf('  - Optimized RBF kernel training and prediction...\n');
    rbf_params = best_params{j};
    mdl_rbf = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params.C, ...
        'KernelScale', 1/sqrt(2*rbf_params.gamma), ...
        'Epsilon', rbf_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j) = predict(mdl_rbf, X_predict_norm);
    
    fprintf('  %s prediction completed\n', output_names{j});
end
fprintf('Optimized RBF SVR prediction completed!\n');

%% 5. Denormalize predictions
fprintf('\nDenormalizing predictions...\n');
Y_predict = zeros(size(Y_predict_norm));
for j = 1:num_outputs
    Y_predict(:,j) = Y_predict_norm(:,j) * Y_std(j) + Y_mean(j);
end
fprintf('Denormalization completed\n');

%% 6. Cross-validation results and performance metrics visualization
fprintf('\n=== Cross-Validation Results Visualization ===\n');

% 6-1. Performance metrics bar chart
figure('Name','RBF SVR K-fold CV Performance Metrics','WindowStyle','docked');

subplot(1,3,1);
bar(CV_scores, 'FaceColor', [0.8 0.2 0.2]); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R² Score');
title('K-fold CV R² Score'); xtickangle(45);
for j = 1:num_outputs
    text(j, CV_scores(j)+0.03, sprintf('%.3f', CV_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

subplot(1,3,2);
bar(rmse_scores, 'FaceColor', [0.8 0.4 0.2]); grid on;
set(gca,'XTickLabel',output_names); ylabel('RMSE (normalized)');
title('K-fold CV RMSE'); xtickangle(45);
for j = 1:num_outputs
    text(j, rmse_scores(j)+max(rmse_scores)*0.03, sprintf('%.3f', rmse_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

subplot(1,3,3);
bar(mae_scores, 'FaceColor', [0.6 0.2 0.8]); grid on;
set(gca,'XTickLabel',output_names); ylabel('MAE (normalized)');
title('K-fold CV MAE'); xtickangle(45);
for j = 1:num_outputs
    text(j, mae_scores(j)+max(mae_scores)*0.03, sprintf('%.3f', mae_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

sgtitle('Optimized RBF SVR K-fold Cross-Validation Performance Metrics', 'FontSize', 14, 'FontWeight', 'bold');

% 6-2. Performance metrics table output
fprintf('\n=== RBF SVR Performance Metrics Summary ===\n');
fprintf('%-12s%-15s%-15s%-15s\n', 'Output', 'CV_R²', 'CV_RMSE', 'CV_MAE');
fprintf(repmat('-', 1, 12 + 15*3));
fprintf('\n');

for j = 1:num_outputs
    fprintf('%-12s%-15.4f%-15.4f%-15.4f\n', output_names{j}, CV_scores(j), rmse_scores(j), mae_scores(j));
end

% Optimal parameter output
fprintf('\n=== Optimal Hyperparameters ===\n');
for j = 1:num_outputs
    rbf_params = best_params{j};
    fprintf('%s: C=%.4f, gamma=%.4f, epsilon=%.4f\n', ...
        output_names{j}, rbf_params.C, rbf_params.gamma, rbf_params.epsilon);
end

%% 7. Prediction visualization
fprintf('\n=== Optimized RBF Visualization Generation ===\n');

% 7-1. Boxplot: RBF predictions vs actual
fprintf('Generating boxplots...\n');
figure('Name','Optimized RBF Prediction Distribution Boxplot: Untested(112) vs Actual(16)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'Actual (16)'},size(Y_train,1),1); repmat({'RBF Prediction (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('Optimized RBF SVR: Actual(16) vs Untested(112) Prediction Distribution Comparison','FontSize',14,'FontWeight','bold');

% 7-2. Scatter plot: Actual value range and predicted values by RBF prediction condition
fprintf('Generating scatter plots and histograms...\n');
figure('Name','Optimized RBF Prediction Scatter Plot and Histogram: Predicted Value Distribution by Condition','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'r', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','Min Actual'); yline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; ylabel(output_names{j}); legend('Optimized RBF SVR','location','best');
    title(['Untested 112 Conditions ',output_names{j},' Optimized RBF Prediction Distribution']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j),'FaceColor','r','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','Min Actual'); xline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('Optimized RBF SVR','location','best');
    title(['Untested 112 Conditions ',output_names{j},' Optimized RBF Prediction Histogram']);
end
sgtitle('Optimized RBF SVR Output Prediction Scatter Plot and Distribution by Condition (16 Actual Value Reference Lines)','FontSize',14,'FontWeight','bold');

% 7-3. Heatmap visualization (Optimized RBF)
fprintf('Generating heatmap...\n');
figure('Name','Optimized RBF Prediction Heatmap','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' Optimized RBF Prediction Heatmap']);
    xlabel('X3*X4 Condition Index'); ylabel('X1*X2 Condition Index');
end
sgtitle('Optimized RBF SVR-Based Prediction Heatmap','FontSize',14);

%% 8. Result storage
fprintf('\n=== Results Saving ===\n');

% Save optimized RBF predictions
fprintf('Saving optimized RBF SVR predictions to predict_rbf_hyper.csv...\n');
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_rbf_hyper.csv');
fprintf('Saving completed: predict_rbf_hyper.csv (%d prediction conditions)\n', size(X_predict,1));

% Save optimal hyperparameters
fprintf('Saving optimal hyperparameters to hyperparams_Augment_2rbf.mat...\n');
save('hyperparams_Augment_2rbf.mat', 'best_params', 'CV_scores', 'rmse_scores', 'mae_scores', ...
     'output_names');
fprintf('Saving completed: hyperparams_Augment_2rbf.mat\n');

fprintf('\n=== Augment_2rbf_hyper.m Execution Completed ===\n');
fprintf('Key Improvements:\n');
fprintf('1. RBF SVR automatic hyperparameter optimization (Grid Search: C, gamma, epsilon)\n');
fprintf('2. Improved performance evaluation with 5-fold cross-validation\n');
fprintf('3. Added RMSE, MAE metrics\n');
fprintf('4. Automatic saving and reuse of optimal parameters\n');
fprintf('5. Enhanced visualization and detailed performance analysis\n');