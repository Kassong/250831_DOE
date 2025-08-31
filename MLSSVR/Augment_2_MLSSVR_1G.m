clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Augment_2_MLSSVR_1G.m: Single Group MLSSVR Model Prediction Started ===\n');
fprintf('Loading data...\n');
T = readtable('dataset.csv');
X_train = T{:,1:4}; Y_train = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
num_outputs = size(Y_train,2);
fprintf('Training data: %d samples, %d output variables\n', size(X_train,1), num_outputs);

%% 1. Taguchi OA Full Condition Generation
fprintf('\nGenerating Taguchi orthogonal array full conditions...\n');
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

%% 2. Input and Output Normalization (Based on Training Set)
fprintf('\nNormalizing data...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_predict_norm = (X_predict - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
fprintf('Normalization completed (based on training set)\n');

%% 3. LOOCV Cross-validation (Single Group MLSSVR)
fprintf('\n=== Single MLSSVR Model (Full Output Group) LOOCV Cross-validation Started ===\n');
n_samples = size(X_train_norm, 1);

% MLSSVR hyperparameter grid search (full data, multi-target simultaneous optimization)
fprintf('Performing hyperparameter grid search...\n');
[gamma_opt, lambda_opt, p_opt, MSE_opt] = GridMLSSVR(X_train_norm, Y_train_norm, 5);
fprintf('Optimal parameters: gamma=%.4f, lambda=%.4f, p=%.4f (MSE=%.6f)\n', gamma_opt, lambda_opt, p_opt, MSE_opt);

% Perform LOOCV
fprintf('Performing MLSSVR LOOCV validation...\n');
loocv_pred = zeros(n_samples, num_outputs);

for i = 1:n_samples
    fprintf('  LOOCV %d/%d...\n', i, n_samples);
    
    % Leave one out
    X_loo_train = X_train_norm([1:i-1, i+1:end], :);
    Y_loo_train = Y_train_norm([1:i-1, i+1:end], :);
    X_loo_test = X_train_norm(i, :);
    Y_loo_test = Y_train_norm(i, :); % 더미용
    
    % Single MLSSVR model training
    [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train, gamma_opt, lambda_opt, p_opt);
    
    % Prediction
    [pred_loo, ~, ~] = MLSSVRPredict(X_loo_test, Y_loo_test, X_loo_train, alpha_loo, b_loo, lambda_opt, p_opt);
    loocv_pred(i, :) = pred_loo;
end

% LOOCV R² 계산
LOOCV_scores = zeros(num_outputs, 1);
for j = 1:num_outputs
    SS_res = sum((Y_train_norm(:,j) - loocv_pred(:,j)).^2);
    SS_tot = sum((Y_train_norm(:,j) - mean(Y_train_norm(:,j))).^2);
    LOOCV_scores(j) = 1 - SS_res/SS_tot;
    fprintf('  %s LOOCV R² = %.4f\n', output_names{j}, LOOCV_scores(j));
end
fprintf('MLSSVR LOOCV cross-validation completed!\n');

%% 4. Single Group MLSSVR Model Training and Prediction
fprintf('\n=== Single MLSSVR Model (Full Output Group) Training and Prediction Started ===\n');

% MLSSVR training (single model construction with full training data)
fprintf('Training MLSSVR model...\n');
[alpha, b] = MLSSVRTrain(X_train_norm, Y_train_norm, gamma_opt, lambda_opt, p_opt);

% Prediction of untested conditions
fprintf('Predicting untested conditions...\n');
Y_predict_dummy = zeros(size(X_predict,1), num_outputs); % Dummy labels
[Y_predict_norm, ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy, X_train_norm, alpha, b, lambda_opt, p_opt);
fprintf('MLSSVR prediction completed!\n');

%% 5. Denormalization of Predictions
fprintf('\nDenormalizing predictions...\n');
Y_predict = zeros(size(Y_predict_norm));
for j = 1:num_outputs
    Y_predict(:,j) = Y_predict_norm(:,j) * Y_std(j) + Y_mean(j);
end
fprintf('Denormalization completed\n');

%% 6. LOOCV Results Visualization
fprintf('\n=== LOOCV Results Visualization ===\n');
figure('Name','Single MLSSVR LOOCV Cross-validation Results','WindowStyle','docked');
bar(LOOCV_scores); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('LOOCV R²');
title('Single MLSSVR Model LOOCV Cross-validation Performance'); xtickangle(45);
for j = 1:num_outputs
    text(j, LOOCV_scores(j)+0.05, sprintf('%.3f', LOOCV_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% 7. Prediction Results Visualization
fprintf('\n=== Generating Visualizations ===\n');
% 7-1. Boxplot: MLSSVR Prediction vs Actual
fprintf('Generating boxplot...\n');
figure('Name','MLSSVR Prediction Distribution Boxplot: Untested(112) vs Actual(16)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'Actual (16)'},size(Y_train,1),1); repmat({'MLSSVR Prediction (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('Single MLSSVR: Actual(16) vs Untested(112) Prediction Distribution Comparison','FontSize',14,'FontWeight','bold');

% 7-2. Scatter plot: MLSSVR prediction by condition with actual value range and predicted values
fprintf('Generating scatter plots and histograms...\n');
figure('Name','MLSSVR Prediction Scatter Plot and Histogram: Prediction Value Distribution by Condition','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'b', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','Min Actual'); yline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; ylabel(output_names{j}); legend('MLSSVR Prediction','location','best');
    title(['Untested 112 Conditions ',output_names{j},' MLSSVR Prediction Distribution']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j),'FaceColor','b','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','Min Actual'); xline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('MLSSVR Prediction','location','best');
    title(['Untested 112 Conditions ',output_names{j},' MLSSVR Prediction Histogram']);
end
sgtitle('Single MLSSVR Condition-wise Output Prediction Scatter Plot and Distribution (16 Actual Value Reference Lines)','FontSize',14,'FontWeight','bold');

% 7-3. Heatmap Visualization (MLSSVR)
fprintf('Generating heatmap...\n');
figure('Name','MLSSVR Prediction Value Heatmap','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' MLSSVR Prediction Heatmap']);
    xlabel('X3*X4 Condition Index'); ylabel('X1*X2 Condition Index');
end
sgtitle('Single MLSSVR-based Prediction Value Heatmap','FontSize',14);

%% 8. Single Group MLSSVR Prediction Values CSV Save
fprintf('\n=== Saving Results ===\n');
fprintf('Saving single group MLSSVR model predictions to predict_MLSSVR_1G.csv...\n');
% Save MLSSVR predictions
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_MLSSVR_1G.csv');
fprintf('Save completed: predict_MLSSVR_1G.csv (%d prediction conditions)\n', size(X_predict,1));
fprintf('\n=== Augment_2_MLSSVR_1G.m Execution Completed ===\n');