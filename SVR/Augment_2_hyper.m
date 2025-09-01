clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Augment_2_hyper.m Untested Condition Prediction (Hyperparameter Optimization + K-fold CV) ===\n');
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

%% 3. K-fold cross-validation and hyperparameter optimization
fprintf('\n=== K-fold Cross-Validation and Hyperparameter Optimization Start ===\n');
kernel_names = {'RBF','Linear','Ensemble'};
k_folds = 5; % 5-fold CV (16개 샘플이므로)
n_samples = size(X_train_norm, 1);

% Hyperparameter search range settings
c_range = logspace(-2, 2, 10); % C: 0.01 ~ 100
gamma_range = logspace(-4, 1, 10); % gamma: 0.0001 ~ 10 (RBF용)
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1

% Generate CV partitions (custom implementation to avoid Bioinformatics Toolbox dependency)
indices = randperm(n_samples);
fold_size = floor(n_samples / k_folds);
cv_indices = zeros(n_samples, 1);
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
best_params = cell(num_outputs, 3); % Optimal parameters for each output-kernel
CV_scores = zeros(num_outputs, 3); % Optimal CV scores for each output-kernel
rmse_scores = zeros(num_outputs, 3);
mae_scores = zeros(num_outputs, 3);

for j = 1:num_outputs
    fprintf('\nOptimizing output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % 1) RBF kernel optimization
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
    
    best_params{j, 1} = best_rbf_params;
    CV_scores(j, 1) = best_score;
    fprintf('    Optimal RBF parameters: C=%.4f, gamma=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_rbf_params.C, best_rbf_params.gamma, best_rbf_params.epsilon, best_score);
    
    % 2) Linear kernel optimization
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
    
    best_params{j, 2} = best_linear_params;
    CV_scores(j, 2) = best_score;
    fprintf('    Optimal Linear parameters: C=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_linear_params.C, best_linear_params.epsilon, best_score);
    
    % 3) Ensemble optimization (optimize each component then combine)
    fprintf('  - Ensemble model optimization...\n');
    
    % Linear component (use optimal parameters from above)
    best_ensemble_linear = best_linear_params;
    
    % Polynomial component optimization
    best_score_poly = -inf;
    best_poly_params = struct();
    poly_orders = [2, 3]; % Polynomial orders
    
    for c_idx = 1:length(c_range)
        for e_idx = 1:length(epsilon_range)
            for p_idx = 1:length(poly_orders)
                % K-fold CV
                cv_predictions = zeros(n_samples, 1);
                
                for fold = 1:k_folds
                    train_idx = cv_indices ~= fold;
                    test_idx = cv_indices == fold;
                    
                    mdl = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
                        'KernelFunction', 'polynomial', 'BoxConstraint', c_range(c_idx), ...
                        'PolynomialOrder', poly_orders(p_idx), ...
                        'Epsilon', epsilon_range(e_idx), 'Standardize', false);
                    
                    cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
                end
                
                % R² calculation
                SS_res = sum((Yt_train - cv_predictions).^2);
                SS_tot = sum((Yt_train - mean(Yt_train)).^2);
                r2_score = 1 - SS_res/SS_tot;
                
                if r2_score > best_score_poly
                    best_score_poly = r2_score;
                    best_poly_params.C = c_range(c_idx);
                    best_poly_params.epsilon = epsilon_range(e_idx);
                    best_poly_params.order = poly_orders(p_idx);
                end
            end
        end
    end
    
    % Ensemble CV evaluation
    cv_predictions_ensemble = zeros(n_samples, 1);
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        % Linear model
        mdl_lin = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
            'KernelFunction', 'linear', 'BoxConstraint', best_ensemble_linear.C, ...
            'Epsilon', best_ensemble_linear.epsilon, 'Standardize', false);
        pred_lin = predict(mdl_lin, X_train_norm(test_idx,:));
        
        % Polynomial model
        mdl_poly = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
            'KernelFunction', 'polynomial', 'BoxConstraint', best_poly_params.C, ...
            'PolynomialOrder', best_poly_params.order, ...
            'Epsilon', best_poly_params.epsilon, 'Standardize', false);
        pred_poly = predict(mdl_poly, X_train_norm(test_idx,:));
        
        cv_predictions_ensemble(test_idx) = (pred_lin + pred_poly) / 2;
    end
    
    % Ensemble R² calculation
    SS_res = sum((Yt_train - cv_predictions_ensemble).^2);
    SS_tot = sum((Yt_train - mean(Yt_train)).^2);
    ensemble_r2 = 1 - SS_res/SS_tot;
    
    best_params{j, 3} = struct('linear', best_ensemble_linear, 'poly', best_poly_params);
    CV_scores(j, 3) = ensemble_r2;
    
    fprintf('    Optimal Ensemble parameters:\n');
    fprintf('      Linear: C=%.4f, epsilon=%.4f\n', best_ensemble_linear.C, best_ensemble_linear.epsilon);
    fprintf('      Poly: C=%.4f, epsilon=%.4f, order=%d\n', best_poly_params.C, best_poly_params.epsilon, best_poly_params.order);
    fprintf('      Ensemble CV R²=%.4f\n', ensemble_r2);
    
    % Calculate RMSE and MAE (for each kernel)
    for k = 1:3
        if k == 1 % RBF
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
        elseif k == 2 % Linear
            cv_pred = zeros(n_samples, 1);
            for fold = 1:k_folds
                train_idx = cv_indices ~= fold;
                test_idx = cv_indices == fold;
                
                mdl = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
                    'KernelFunction', 'linear', 'BoxConstraint', best_linear_params.C, ...
                    'Epsilon', best_linear_params.epsilon, 'Standardize', false);
                cv_pred(test_idx) = predict(mdl, X_train_norm(test_idx,:));
            end
        else % Ensemble
            cv_pred = cv_predictions_ensemble;
        end
        
        rmse_scores(j, k) = sqrt(mean((Yt_train - cv_pred).^2));
        mae_scores(j, k) = mean(abs(Yt_train - cv_pred));
    end
end

fprintf('\nK-fold cross-validation and hyperparameter optimization completed!\n');

%% 4. Perform predictions with optimized models
fprintf('\n=== Optimized SVR Model Training and Prediction Start ===\n');
Y_predict_norm = zeros(size(X_predict,1), num_outputs, 3);

for j = 1:num_outputs
    fprintf('Predicting output variable %s (%d/%d)...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % RBF (using optimal parameters)
    fprintf('  - Optimized RBF kernel training and prediction...\n');
    rbf_params = best_params{j, 1};
    mdl_rbf = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params.C, ...
        'KernelScale', 1/sqrt(2*rbf_params.gamma), ...
        'Epsilon', rbf_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j,1) = predict(mdl_rbf, X_predict_norm);
    
    % Linear (using optimal parameters)
    fprintf('  - Optimized Linear kernel training and prediction...\n');
    linear_params = best_params{j, 2};
    mdl_lin = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j,2) = predict(mdl_lin, X_predict_norm);
    
    % Ensemble (using optimal parameters)
    fprintf('  - Optimized Ensemble model training and prediction...\n');
    ensemble_params = best_params{j, 3};
    
    mdl_lin_ens = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', ensemble_params.linear.C, ...
        'Epsilon', ensemble_params.linear.epsilon, 'Standardize', false);
    
    mdl_poly_ens = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'polynomial', ...
        'BoxConstraint', ensemble_params.poly.C, ...
        'PolynomialOrder', ensemble_params.poly.order, ...
        'Epsilon', ensemble_params.poly.epsilon, 'Standardize', false);
    
    Y_pred_lin_ens = predict(mdl_lin_ens, X_predict_norm);
    Y_pred_poly_ens = predict(mdl_poly_ens, X_predict_norm);
    Y_predict_norm(:,j,3) = (Y_pred_lin_ens + Y_pred_poly_ens) / 2;
    
    fprintf('  %s prediction completed\n', output_names{j});
end
fprintf('All model predictions completed!\n');

%% 5. Denormalize predictions
fprintf('\nDenormalizing predictions...\n');
Y_predict = zeros(size(Y_predict_norm));
for k = 1:3
    for j = 1:num_outputs
        Y_predict(:,j,k) = Y_predict_norm(:,j,k) * Y_std(j) + Y_mean(j);
    end
end
fprintf('Denormalization completed\n');

%% 6. Cross-validation results and performance metrics visualization
fprintf('\n=== Cross-Validation Results Visualization ===\n');

% 6-1. R² scores
figure('Name','K-fold CV Performance Metrics','WindowStyle','docked');
subplot(1,3,1);
for j = 1:num_outputs
    plot(1:3, CV_scores(j,:), 'o-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
end
set(gca, 'XTick', 1:3, 'XTickLabel', kernel_names);
ylabel('R² Score'); title('K-fold CV R² Score');
legend(output_names, 'Location', 'best'); grid on;

% 6-2. RMSE
subplot(1,3,2);
for j = 1:num_outputs
    plot(1:3, rmse_scores(j,:), 's-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
end
set(gca, 'XTick', 1:3, 'XTickLabel', kernel_names);
ylabel('RMSE (normalized)'); title('K-fold CV RMSE');
legend(output_names, 'Location', 'best'); grid on;

% 6-3. MAE
subplot(1,3,3);
for j = 1:num_outputs
    plot(1:3, mae_scores(j,:), '^-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
end
set(gca, 'XTick', 1:3, 'XTickLabel', kernel_names);
ylabel('MAE (normalized)'); title('K-fold CV MAE');
legend(output_names, 'Location', 'best'); grid on;

sgtitle('K-fold Cross-Validation Performance Metrics (Optimized Hyperparameters)', 'FontSize', 14, 'FontWeight', 'bold');

% 6-4. Performance metrics table output
fprintf('\n=== Performance Metrics Summary ===\n');
fprintf('%-12s', 'Output');
for k = 1:3
    fprintf('%-15s%-15s%-15s', [kernel_names{k} '_R²'], [kernel_names{k} '_RMSE'], [kernel_names{k} '_MAE']);
end
fprintf('\n');
fprintf(repmat('-', 1, 12 + 15*3*3));
fprintf('\n');

for j = 1:num_outputs
    fprintf('%-12s', output_names{j});
    for k = 1:3
        fprintf('%-15.4f%-15.4f%-15.4f', CV_scores(j,k), rmse_scores(j,k), mae_scores(j,k));
    end
    fprintf('\n');
end

%% 7. 예측값 시각화 (기존 코드와 동일)
fprintf('\n=== Visualization Generation ===\n');

% 7-1. Boxplot: 112 predictions (by kernel) vs 16 actual
fprintf('Generating boxplots...\n');
figure('Name','Prediction Distribution Boxplot: Untested(112) vs Actual(16) - Optimized','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j,1); Y_predict(:,j,2); Y_predict(:,j,3)];
    labels_box = [repmat({'Actual'},size(Y_train,1),1); repmat({'RBF'},size(Y_predict,1),1); ...
                  repmat({'Linear'},size(Y_predict,1),1); repmat({'Ensemble'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('Actual(16) vs Untested(112) Prediction Distribution Comparison (Optimized)','FontSize',14,'FontWeight','bold');

% 7-2. Scatter plot: Actual value range and predicted values by prediction condition
fprintf('Generating scatter plots and histograms...\n');
figure('Name','Prediction Scatter Plot and Histogram: Predicted Value Distribution by Condition - Optimized','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j,1), 60, 'r', 'filled'); hold on; % RBF
    scatter(1:size(Y_predict,1), Y_predict(:,j,2), 60, 'g', 'filled');
    scatter(1:size(Y_predict,1), Y_predict(:,j,3), 60, 'b', 'filled');
    yline(min(Y_train(:,j)),'k:','Min Actual'); yline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; ylabel(output_names{j}); legend('RBF','Linear','Ensemble','location','best');
    title(['Untested 112 Conditions ',output_names{j},' Prediction Distribution (Optimized)']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j,1),'FaceColor','r','EdgeAlpha',0.1); hold on;
    histogram(Y_predict(:,j,2),'FaceColor','g','EdgeAlpha',0.1);
    histogram(Y_predict(:,j,3),'FaceColor','b','EdgeAlpha',0.1);
    xline(min(Y_train(:,j)),'k:','Min Actual'); xline(max(Y_train(:,j)),'k:','Max Actual');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('RBF','Linear','Ensemble','location','best');
    title(['Untested 112 Conditions ',output_names{j},' Prediction Histogram (Optimized)']);
end
sgtitle('Output Prediction Scatter Plot and Distribution by Condition (Optimized, 16 Actual Value Reference Lines)','FontSize',14,'FontWeight','bold');

% 7-3. Heatmap visualization
fprintf('Generating heatmap...\n');
figure('Name','Prediction Heatmap (Optimized RBF)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j,1), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' Optimized RBF Prediction Heatmap']);
    xlabel('X3*X4 Condition Index'); ylabel('X1*X2 Condition Index');
end
sgtitle('Optimized RBF SVR-Based Prediction Heatmap','FontSize',14);

%% 8. Save ensemble prediction values to CSV
fprintf('\n=== Results Saving ===\n');
fprintf('Saving optimized ensemble model predictions to predict_hyper.csv...\n');
% Save only ensemble predictions from each model (3rd dimension is ensemble)
ensemble_predictions = Y_predict(:,:,3);
predict_table = array2table([X_predict, ensemble_predictions], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_hyper.csv');
fprintf('Saving completed: predict_hyper.csv (%d prediction conditions)\n', size(X_predict,1));

% Save optimal hyperparameters
fprintf('Saving optimal hyperparameters to hyperparams_Augment_2.mat...\n');
save('hyperparams_Augment_2.mat', 'best_params', 'CV_scores', 'rmse_scores', 'mae_scores', ...
     'output_names', 'kernel_names');
fprintf('Saving completed: hyperparams_Augment_2.mat\n');

fprintf('\n=== Augment_2_hyper.m Execution Completed ===\n');
fprintf('Key Improvements:\n');
fprintf('1. Automatic hyperparameter optimization (Grid Search)\n');
fprintf('2. Improved performance evaluation with 5-fold cross-validation\n');
fprintf('3. Added RMSE, MAE metrics\n');
fprintf('4. Automatic saving and reuse of optimal parameters\n');