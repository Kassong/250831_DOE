clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Augment_2_MLSSVR_2G.m: 2-Group MLSSVR Model Prediction Started ===\n');
fprintf('Loading data...\n');
T = readtable('dataset.csv');
X_train = T{:,1:4}; Y_train = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
fprintf('Training data: %d samples, %d total output variables\n', size(X_train,1), size(Y_train,2));

%% 0. Output Variable Grouping
Y_train_g1 = Y_train(:, 1:2); output_names_g1 = output_names(1:2);
Y_train_g2 = Y_train(:, 3:4); output_names_g2 = output_names(3:4);
fprintf('Split output variables into 2 groups: (1,2) / (3,4)\n');

%% 1. Taguchi OA Full Condition Generation
fprintf('\nGenerating Taguchi orthogonal array full conditions...\n');
x1_values = [250, 750, 1250, 1750];
x2_values = [20, 40, 60, 80];
x3_values = [150, 300, 450, 600];
x4_values = [4, 8];
[X1, X2, X3, X4] = ndgrid(x1_values, x2_values, x3_values, x4_values);
X_all = [X1(:), X2(:), X3(:), X4(:)];
fprintf('Total conditions: %d\n', size(X_all,1));

% Extract prediction target conditions excluding training set
is_train = ismember(X_all, X_train, 'rows');
X_predict = X_all(~is_train,:);
fprintf('Prediction targets after excluding training conditions: %d conditions\n', size(X_predict,1));

%% 2. Data Normalization (Based on Training Set)
fprintf('\nNormalizing data...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_predict_norm = (X_predict - X_mean) ./ X_std;
% Group-wise output normalization
[Y_train_norm_g1, Y_mean_g1, Y_std_g1] = zscore(Y_train_g1);
[Y_train_norm_g2, Y_mean_g2, Y_std_g2] = zscore(Y_train_g2);
fprintf('Normalization completed (input and group-wise output)\n');

%% 3. Group-wise LOOCV Cross-validation
fprintf('\n=== Group-wise MLSSVR Model LOOCV Cross-validation Started ===\n');
n_samples = size(X_train_norm, 1);
loocv_pred_g1 = zeros(n_samples, size(Y_train_g1,2));
loocv_pred_g2 = zeros(n_samples, size(Y_train_g2,2));

% --- Group 1 (outputs 1,2) ---
fprintf('\n--- Group 1 (outputs 1,2) validation in progress ---\n');
fprintf('Performing hyperparameter grid search...\n');
[gamma_opt_g1, lambda_opt_g1, p_opt_g1, ~] = GridMLSSVR(X_train_norm, Y_train_norm_g1, 5);
fprintf('Optimal parameters (G1): gamma=%.4f, lambda=%.4f, p=%.4f\n', gamma_opt_g1, lambda_opt_g1, p_opt_g1);
for i = 1:n_samples
    if mod(i,5)==0, fprintf('  G1 LOOCV %d/%d...\n', i, n_samples); end
    X_loo_train = X_train_norm([1:i-1, i+1:end], :); X_loo_test = X_train_norm(i, :);
    Y_loo_train = Y_train_norm_g1([1:i-1, i+1:end], :); Y_loo_test = Y_train_norm_g1(i, :);
    [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train, gamma_opt_g1, lambda_opt_g1, p_opt_g1);
    [pred_loo, ~, ~] = MLSSVRPredict(X_loo_test, Y_loo_test, X_loo_train, alpha_loo, b_loo, lambda_opt_g1, p_opt_g1);
    loocv_pred_g1(i, :) = pred_loo;
end

% --- Group 2 (outputs 3,4) ---
fprintf('\n--- Group 2 (outputs 3,4) validation in progress ---\n');
fprintf('Performing hyperparameter grid search...\n');
[gamma_opt_g2, lambda_opt_g2, p_opt_g2, ~] = GridMLSSVR(X_train_norm, Y_train_norm_g2, 5);
fprintf('Optimal parameters (G2): gamma=%.4f, lambda=%.4f, p=%.4f\n', gamma_opt_g2, lambda_opt_g2, p_opt_g2);
for i = 1:n_samples
    if mod(i,5)==0, fprintf('  G2 LOOCV %d/%d...\n', i, n_samples); end
    X_loo_train = X_train_norm([1:i-1, i+1:end], :); X_loo_test = X_train_norm(i, :);
    Y_loo_train = Y_train_norm_g2([1:i-1, i+1:end], :); Y_loo_test = Y_train_norm_g2(i, :);
    [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train, gamma_opt_g2, lambda_opt_g2, p_opt_g2);
    [pred_loo, ~, ~] = MLSSVRPredict(X_loo_test, Y_loo_test, X_loo_train, alpha_loo, b_loo, lambda_opt_g2, p_opt_g2);
    loocv_pred_g2(i, :) = pred_loo;
end

% LOOCV R² calculation and integration
LOOCV_scores_g1 = 1 - sum((Y_train_norm_g1 - loocv_pred_g1).^2) ./ sum((Y_train_norm_g1 - mean(Y_train_norm_g1)).^2);
LOOCV_scores_g2 = 1 - sum((Y_train_norm_g2 - loocv_pred_g2).^2) ./ sum((Y_train_norm_g2 - mean(Y_train_norm_g2)).^2);
LOOCV_scores = [LOOCV_scores_g1, LOOCV_scores_g2]';
for j=1:length(output_names), fprintf('  %s LOOCV R² = %.4f\n', output_names{j}, LOOCV_scores(j)); end
fprintf('MLSSVR LOOCV cross-validation completed!\n');


%% 4. Group-wise MLSSVR Model Training and Prediction
fprintf('\n=== Group-wise MLSSVR Model Training and Prediction Started ===\n');
% --- Group 1 (outputs 1,2) ---
fprintf('Training and predicting Group 1 model...\n');
[alpha_g1, b_g1] = MLSSVRTrain(X_train_norm, Y_train_norm_g1, gamma_opt_g1, lambda_opt_g1, p_opt_g1);
Y_predict_dummy_g1 = zeros(size(X_predict,1), size(Y_train_g1,2));
[Y_predict_norm_g1, ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy_g1, X_train_norm, alpha_g1, b_g1, lambda_opt_g1, p_opt_g1);

% --- Group 2 (outputs 3,4) ---
fprintf('Training and predicting Group 2 model...\n');
[alpha_g2, b_g2] = MLSSVRTrain(X_train_norm, Y_train_norm_g2, gamma_opt_g2, lambda_opt_g2, p_opt_g2);
Y_predict_dummy_g2 = zeros(size(X_predict,1), size(Y_train_g2,2));
[Y_predict_norm_g2, ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy_g2, X_train_norm, alpha_g2, b_g2, lambda_opt_g2, p_opt_g2);
fprintf('Group-wise prediction completed!\n');

%% 5. Denormalization and Integration of Predictions
fprintf('\nDenormalizing predictions...\n');
Y_predict_g1 = Y_predict_norm_g1 .* Y_std_g1 + Y_mean_g1;
Y_predict_g2 = Y_predict_norm_g2 .* Y_std_g2 + Y_mean_g2;
Y_predict = [Y_predict_g1, Y_predict_g2]; % Final prediction integration
fprintf('Denormalization and result integration completed\n');

%% 6. LOOCV Results Visualization
fprintf('\n=== LOOCV Results Visualization ===\n');
figure('Name','2-Group MLSSVR LOOCV Cross-validation Results','WindowStyle','docked');
bar(LOOCV_scores); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('LOOCV R²');
title('2-Group MLSSVR Model LOOCV Cross-validation Performance'); xtickangle(45);
for j = 1:length(LOOCV_scores)
    text(j, LOOCV_scores(j)+0.05, sprintf('%.3f', LOOCV_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% 7. Prediction Results Visualization (same as before)
fprintf('\n=== Generating Visualizations ===\n');
% 7-1. Boxplot
figure('Name','MLSSVR Prediction Distribution Boxplot: Untested(112) vs Actual(16)','WindowStyle','docked');
for j = 1:length(output_names)
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'Actual (16)'},size(Y_train,1),1); repmat({'MLSSVR Prediction (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('2-Group MLSSVR: Actual(16) vs Untested(112) Prediction Distribution Comparison','FontSize',14,'FontWeight','bold');

% 7-2. Scatter plot and histogram
figure('Name','MLSSVR Prediction Scatter Plot and Histogram: Prediction Value Distribution by Condition','WindowStyle','docked');
for j = 1:length(output_names)
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'b', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','Min Actual'); yline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; ylabel(output_names{j}); legend('MLSSVR Prediction','location','best');
    title(['Untested 112 Conditions ',output_names{j},' MLSSVR Prediction Distribution']);
    subplot(4,2,j+length(output_names));
    histogram(Y_predict(:,j),'FaceColor','b','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','Min Actual'); xline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('MLSSVR Prediction','location','best');
    title(['Untested 112 Conditions ',output_names{j},' MLSSVR Prediction Histogram']);
end
sgtitle('2-Group MLSSVR Condition-wise Output Prediction Scatter Plot and Distribution (16 Actual Value Reference Lines)','FontSize',14,'FontWeight','bold');

% 7-3. Heatmap
figure('Name','MLSSVR Prediction Value Heatmap','WindowStyle','docked');
for j = 1:length(output_names)
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' MLSSVR Prediction Heatmap']);
    xlabel('X3*X4 Condition Index'); ylabel('X1*X2 Condition Index');
end
sgtitle('2-Group MLSSVR-based Prediction Value Heatmap','FontSize',14);

%% 8. 2-Group MLSSVR Prediction Values CSV Save
fprintf('\n=== Saving Results ===\n');
fprintf('Saving 2-Group MLSSVR model predictions to predict_MLSSVR_2G.csv...\n');
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_MLSSVR_2G.csv');
fprintf('Save completed: predict_MLSSVR_2G.csv (%d prediction conditions)\n', size(X_predict,1));
fprintf('\n=== Augment_2_MLSSVR_2G.m Execution Completed ===\n');