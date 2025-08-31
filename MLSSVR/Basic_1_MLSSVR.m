clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Basic_1_MLSSVR.m MLSSVR Model Training and Evaluation Started ===\n');
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
fprintf('Data split completed: %d training, %d testing\n', N_train, length(test_idx));

% Normalization (based on training set)
fprintf('Normalizing data...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_test_norm = (X_test - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
Y_test_norm = (Y_test - Y_mean) ./ Y_std;
fprintf('Data normalization completed\n');

%% 2. MLSSVR Model Training and Prediction
fprintf('\n=== MLSSVR Model Training Started ===\n');

% MLSSVR hyperparameter configuration
fprintf('Performing hyperparameter grid search...\n');
[gamma_opt, lambda_opt, p_opt, MSE_opt] = GridMLSSVR(X_train_norm, Y_train_norm, 5);
fprintf('Optimal parameters: gamma=%.4f, lambda=%.4f, p=%.4f (MSE=%.6f)\n', gamma_opt, lambda_opt, p_opt, MSE_opt);

% MLSSVR training
fprintf('Training MLSSVR model...\n');
[alpha, b] = MLSSVRTrain(X_train_norm, Y_train_norm, gamma_opt, lambda_opt, p_opt);

% Training data prediction
fprintf('Predicting training data...\n');
[Y_pred_train_norm, ~, ~] = MLSSVRPredict(X_train_norm, Y_train_norm, X_train_norm, alpha, b, lambda_opt, p_opt);

% Test data prediction
fprintf('Predicting test data...\n');
[Y_pred_test_norm, ~, ~] = MLSSVRPredict(X_test_norm, Y_test_norm, X_train_norm, alpha, b, lambda_opt, p_opt);

% R² 계산
R2_train = zeros(num_outputs, 1);
R2_test = zeros(num_outputs, 1);
for j = 1:num_outputs
    % Training R²
    SS_res_train = sum((Y_train_norm(:,j) - Y_pred_train_norm(:,j)).^2);
    SS_tot_train = sum((Y_train_norm(:,j) - mean(Y_train_norm(:,j))).^2);
    R2_train(j) = 1 - SS_res_train/SS_tot_train;
    
    % Test R²
    SS_res_test = sum((Y_test_norm(:,j) - Y_pred_test_norm(:,j)).^2);
    SS_tot_test = sum((Y_test_norm(:,j) - mean(Y_test_norm(:,j))).^2);
    R2_test(j) = 1 - SS_res_test/SS_tot_test;
    
    fprintf('  %s: R²_train=%.3f, R²_test=%.3f\n', output_names{j}, R2_train(j), R2_test(j));
end
fprintf('MLSSVR model training completed!\n');

%% 3. Denormalization
fprintf('\nDenormalizing predictions...\n');
Y_pred_train_denorm = zeros(size(Y_pred_train_norm));
Y_pred_test_denorm = zeros(size(Y_pred_test_norm));

for j = 1:num_outputs
    Y_pred_train_denorm(:,j) = Y_pred_train_norm(:,j) * Y_std(j) + Y_mean(j);
    Y_pred_test_denorm(:,j)  = Y_pred_test_norm(:,j)  * Y_std(j) + Y_mean(j);
end
fprintf('Denormalization completed\n');

%% 4. MLSSVR Performance Output
fprintf('\n=== MLSSVR Performance Summary ===\n');
Yfit_train_best = Y_pred_train_denorm;
Yfit_test_best  = Y_pred_test_denorm;
for j = 1:num_outputs
    fprintf('%s: MLSSVR R²_train=%.3f, R²_test=%.3f\n', output_names{j}, R2_train(j), R2_test(j));
end

%% 5. MLSSVR R² Bar Plot
fprintf('\nGenerating visualizations...\n');
figure('Name','MLSSVR R²','WindowStyle','docked');
subplot(1,2,1);
bar(R2_train); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R² (training)');
title('MLSSVR Training R²'); xtickangle(45);

subplot(1,2,2);
bar(R2_test); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R² (testing)');
title('MLSSVR Testing R²'); xtickangle(45);
sgtitle('MLSSVR Performance','FontSize',14,'FontWeight','bold');

%% 6. Predicted vs Actual CORRELATION PLOT (Training/Testing)
figure('Name','MLSSVR Correlation Plots','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    scatter(Yfit_test_best(:,j), Y_test(:,j), 120, 'r', 'filled'); hold on; % TEST
    scatter(Yfit_train_best(:,j), Y_train(:,j), 80, 'b', 'filled','MarkerFaceAlpha',0.5); % TRAIN
    xls = min([Y_test(:,j); Yfit_test_best(:,j); Y_train(:,j); Yfit_train_best(:,j)]);
    xhs = max([Y_test(:,j); Yfit_test_best(:,j); Y_train(:,j); Yfit_train_best(:,j)]);
    plot([xls xhs],[xls xhs],'k--','LineWidth',1.5); % 1:1 reference line
    grid on; axis equal; box on;
    xlabel('MLSSVR Predicted Values');
    ylabel('Actual Values');
    title(sprintf('Output: %s\nMLSSVR R²_{train}=%.3f, R²_{test}=%.3f', ...
        output_names{j}, R2_train(j), R2_test(j)));
    legend('Test Prediction','Training Prediction','1:1 Reference Line','Location','best');
end
sgtitle('MLSSVR Training/Testing Correlation Plot','FontSize',14,'FontWeight','bold');

%% 7. Boxplot Distribution Comparison (Training/Testing/Prediction)
figure('Name','MLSSVR Prediction Distribution Boxplot','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Yfit_train_best(:,j); Y_test(:,j); Yfit_test_best(:,j)];
    labels_box = [repmat({'Training Actual'},N_train,1); repmat({'Training Predicted'},N_train,1); ...
                  repmat({'Testing Actual'},num_samples-N_train,1); repmat({'Testing Predicted'},num_samples-N_train,1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('MLSSVR Distribution Comparison (Training/Testing/Prediction)','FontSize',14,'FontWeight','bold');
