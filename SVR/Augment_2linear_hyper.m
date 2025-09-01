clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Augment_2linear_hyper.m Linear SVR Untested Condition Prediction (Hyperparameter Optimization + K-fold CV) ===\n');
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

%% 3. Linear SVR hyperparameter optimization and K-fold cross-validation
fprintf('\n=== Linear SVR Hyperparameter Optimization and K-fold CV Start ===\n');

% Hyperparameter search range settings
c_range = logspace(-2, 2, 10); % C: 0.01 ~ 100
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
    fprintf('\nOptimizing Linear SVR for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % Linear kernel optimization
    fprintf('  - Linear kernel hyperparameter optimization...\n');
    best_score = -inf;
    best_linear_params = struct();
    
    for c_idx = 1:length(c_range)
        for e_idx = 1:length(epsilon_range)
            % K-fold CV
            cv_predictions = zeros(n_samples, 1);
            
            for fold = 1:k_folds
                train_idx = cv_indices ~= fold;
                test_idx = cv_indices == fold;
                
                mdl = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
                    'KernelFunction', 'linear', 'BoxConstraint', c_range(c_idx), ...
                    'Epsilon', epsilon_range(e_idx), 'Standardize', false);
                
                cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
            end
            
            % R² calculation
            SS_res = sum((Yt_train - cv_predictions).^2);
            SS_tot = sum((Yt_train - mean(Yt_train)).^2);
            r2_score = 1 - SS_res/SS_tot;
            
            if r2_score > best_score
                best_score = r2_score;
                best_linear_params.C = c_range(c_idx);
                best_linear_params.epsilon = epsilon_range(e_idx);
            end
        end
    end
    
    best_params{j} = best_linear_params;
    CV_scores(j) = best_score;
    fprintf('    Optimal Linear parameters: C=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_linear_params.C, best_linear_params.epsilon, best_score);
    
    % Calculate RMSE and MAE (with optimal parameters)
    cv_pred = zeros(n_samples, 1);
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        mdl = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
            'KernelFunction', 'linear', 'BoxConstraint', best_linear_params.C, ...
            'Epsilon', best_linear_params.epsilon, 'Standardize', false);
        cv_pred(test_idx) = predict(mdl, X_train_norm(test_idx,:));
    end
    
    rmse_scores(j) = sqrt(mean((Yt_train - cv_pred).^2));
    mae_scores(j) = mean(abs(Yt_train - cv_pred));
    
    fprintf('    CV RMSE=%.4f, MAE=%.4f\n', rmse_scores(j), mae_scores(j));
end

fprintf('\nLinear SVR hyperparameter optimization completed!\n');

%% 4. Optimized Linear SVR model training and prediction
fprintf('\n=== Optimized Linear SVR Model Training and Prediction Start ===\n');
Y_predict_norm = zeros(size(X_predict,1), num_outputs);

for j = 1:num_outputs
    fprintf('Predicting optimized Linear for output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % Optimized Linear SVR
    fprintf('  - Optimized Linear kernel training and prediction...\n');
    linear_params = best_params{j};
    mdl_lin = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j) = predict(mdl_lin, X_predict_norm);
    
    fprintf('  %s prediction completed\n', output_names{j});
end
fprintf('Optimized Linear SVR prediction completed!\n');

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
figure('Name','Linear SVR K-fold CV Performance Metrics','WindowStyle','docked');

subplot(1,3,1);
bar(CV_scores, 'FaceColor', [0.2 0.6 0.8]); ylim([0 1]); grid on;
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
bar(mae_scores, 'FaceColor', [0.4 0.8 0.2]); grid on;
set(gca,'XTickLabel',output_names); ylabel('MAE (normalized)');
title('K-fold CV MAE'); xtickangle(45);
for j = 1:num_outputs
    text(j, mae_scores(j)+max(mae_scores)*0.03, sprintf('%.3f', mae_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

sgtitle('Optimized Linear SVR K-fold Cross-Validation Performance Metrics', 'FontSize', 14, 'FontWeight', 'bold');

% 6-2. Performance metrics table output
fprintf('\n=== Linear SVR Performance Metrics Summary ===\n');
fprintf('%-12s%-15s%-15s%-15s\n', 'Output', 'CV_R²', 'CV_RMSE', 'CV_MAE');
fprintf(repmat('-', 1, 12 + 15*3));
fprintf('\n');

for j = 1:num_outputs
    fprintf('%-12s%-15.4f%-15.4f%-15.4f\n', output_names{j}, CV_scores(j), rmse_scores(j), mae_scores(j));
end

% Optimal parameter output
fprintf('\n=== Optimal Hyperparameters ===\n');
for j = 1:num_outputs
    linear_params = best_params{j};
    fprintf('%s: C=%.4f, epsilon=%.4f\n', output_names{j}, linear_params.C, linear_params.epsilon);
end

%% 7. 예측값 시각화
fprintf('\n=== Optimized Linear Visualization Generation ===\n');

% 7-1. Boxplot: Linear predictions vs actual
fprintf('Generating boxplots...\n');
figure('Name','Optimized Linear Prediction Distribution Boxplot: Untested(112) vs Actual(16)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'Actual (16)'},size(Y_train,1),1); repmat({'Linear Prediction (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('Optimized Linear SVR: Actual(16) vs Untested(112) Prediction Distribution Comparison','FontSize',14,'FontWeight','bold');

% 7-2. Scatter plot: Actual value range and predicted values by Linear prediction condition
fprintf('Generating scatter plots and histograms...\n');
figure('Name','Optimized Linear Prediction Scatter Plot and Histogram: Predicted Value Distribution by Condition','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'g', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','Min Actual'); yline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; ylabel(output_names{j}); legend('Optimized Linear SVR','location','best');
    title(['Untested 112 Conditions ',output_names{j},' Optimized Linear Prediction Distribution']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j),'FaceColor','g','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','Min Actual'); xline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('Optimized Linear SVR','location','best');
    title(['Untested 112 Conditions ',output_names{j},' Optimized Linear Prediction Histogram']);
end
sgtitle('Optimized Linear SVR Output Prediction Scatter Plot and Distribution by Condition (16 Actual Value Reference Lines)','FontSize',14,'FontWeight','bold');

% 7-3. Heatmap visualization (Optimized Linear)
fprintf('Generating heatmap...\n');
figure('Name','Optimized Linear Prediction Heatmap','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' Optimized Linear Prediction Heatmap']);
    xlabel('X3*X4 Condition Index'); ylabel('X1*X2 Condition Index');
end
sgtitle('Optimized Linear SVR-Based Prediction Heatmap','FontSize',14);

%% 8. Result storage
fprintf('\n=== Results Saving ===\n');

% Save optimized Linear predictions
fprintf('Saving optimized Linear SVR predictions to predict_linear_hyper.csv...\n');
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_linear_hyper.csv');
fprintf('Saving completed: predict_linear_hyper.csv (%d prediction conditions)\n', size(X_predict,1));

% Save optimal hyperparameters
fprintf('Saving optimal hyperparameters to hyperparams_Augment_2linear.mat...\n');
save('hyperparams_Augment_2linear.mat', 'best_params', 'CV_scores', 'rmse_scores', 'mae_scores', ...
     'output_names');
fprintf('Saving completed: hyperparams_Augment_2linear.mat\n');

fprintf('\n=== Augment_2linear_hyper.m Execution Completed ===\n');
fprintf('Key Improvements:\n');
fprintf('1. Linear SVR automatic hyperparameter optimization (Grid Search)\n');
fprintf('2. Improved performance evaluation with 5-fold cross-validation\n');
fprintf('3. Added RMSE, MAE metrics\n');
fprintf('4. Automatic saving and reuse of optimal parameters\n');
fprintf('5. Enhanced visualization and detailed performance analysis\n');