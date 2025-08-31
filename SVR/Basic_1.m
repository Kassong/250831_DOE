clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Basic_1.m SVR 모델 학습 및 평가 시작 (하이퍼파라미터 최적화 & K-fold CV 포함) ===\n');
fprintf('데이터 로딩 중...\n');
T = readtable('dataset.csv');
X = T{:,1:4}; Y = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
num_outputs = size(Y,2); num_samples = size(X,1);
fprintf('데이터 로딩 완료: %d개 샘플, %d개 입력변수, %d개 출력변수\n', num_samples, length(input_names), num_outputs);

%% 1. Split (80% 학습, 20% 테스트)
fprintf('\n데이터 분할 중...\n');
idx = randperm(num_samples);
N_train = round(0.8*num_samples);
train_idx = idx(1:N_train); test_idx = idx(N_train+1:end);

X_train = X(train_idx,:); X_test = X(test_idx,:);
Y_train = Y(train_idx,:); Y_test = Y(test_idx,:);
fprintf('데이터 분할 완료: 학습 %d개, 테스트 %d개\n', N_train, length(test_idx));

% 정규화 (학습셋 기준)
fprintf('데이터 정규화 중...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_test_norm = (X_test - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
Y_test_norm = (Y_test - Y_mean) ./ Y_std;
fprintf('데이터 정규화 완료\n');

%% 하이퍼파라미터 최적화 함수 정의
function [best_params, best_score] = optimizeSVR(X, Y, kernel_type, cv_folds)
    % SVR 하이퍼파라미터 그리드 서치
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
    
    % K-fold 교차검증
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

%% 2. 커널별 모델 학습 및 예측 (하이퍼파라미터 최적화 포함)
fprintf('\n=== SVR 모델 학습 시작 (하이퍼파라미터 최적화) ===\n');
kernel_names = {'RBF','Linear','Ensemble'};
Y_pred_train_all = zeros(N_train, num_outputs, 3);
Y_pred_test_all  = zeros(num_samples-N_train, num_outputs, 3);

% 성능 지표 저장 변수
R2_train = zeros(num_outputs, 3);
R2_test = zeros(num_outputs, 3);
RMSE_train = zeros(num_outputs, 3);
RMSE_test = zeros(num_outputs, 3);
MAE_train = zeros(num_outputs, 3);
MAE_test = zeros(num_outputs, 3);
CV_scores = zeros(num_outputs, 3);

% K-fold 설정
k_folds = 5;

for j = 1:num_outputs
    fprintf('\n출력변수 %s (%d/%d) 모델 학습 중...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    Yt_test = Y_test_norm(:,j);

    % RBF 커널 (하이퍼파라미터 최적화)
    fprintf('  - RBF 커널 하이퍼파라미터 최적화 중...\n');
    [rbf_params, rbf_cv_score] = optimizeSVR(X_train_norm, Yt_train, 'rbf', k_folds);
    fprintf('    최적 RBF 파라미터: C=%.2f, Sigma=%.2f (CV R²=%.4f)\n', rbf_params(1), rbf_params(2), rbf_cv_score);
    
    mdl_rbf = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params(1), 'KernelScale', rbf_params(2), 'Standardize', false);
    Y_pred_train_all(:,j,1) = predict(mdl_rbf, X_train_norm);
    Y_pred_test_all(:,j,1) = predict(mdl_rbf, X_test_norm);
    CV_scores(j,1) = rbf_cv_score;

    % Linear 커널 (하이퍼파라미터 최적화)
    fprintf('  - Linear 커널 하이퍼파라미터 최적화 중...\n');
    [lin_params, lin_cv_score] = optimizeSVR(X_train_norm, Yt_train, 'linear', k_folds);
    fprintf('    최적 Linear 파라미터: C=%.2f (CV R²=%.4f)\n', lin_params(1), lin_cv_score);
    
    mdl_lin = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', lin_params(1), 'Standardize', false);
    Y_pred_train_all(:,j,2) = predict(mdl_lin, X_train_norm);
    Y_pred_test_all(:,j,2) = predict(mdl_lin, X_test_norm);
    CV_scores(j,2) = lin_cv_score;

    % Ensemble (Linear + Polynomial, 각각 최적화)
    fprintf('  - Ensemble 모델 (Linear+Poly2) 하이퍼파라미터 최적화 중...\n');
    [poly_params, poly_cv_score] = optimizeSVR(X_train_norm, Yt_train, 'polynomial', k_folds);
    fprintf('    최적 Polynomial 파라미터: C=%.2f, Order=%d (CV R²=%.4f)\n', poly_params(1), poly_params(2), poly_cv_score);
    
    % 최적화된 파라미터로 앙상블 모델 학습
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
    CV_scores(j,3) = (lin_cv_score + poly_cv_score)/2; % 앙상블 CV 점수
    
    % 성능 지표 계산 (모든 커널에 대해)
    for k = 1:3
        % R² 계산
        R2_train(j,k) = 1 - sum((Yt_train - Y_pred_train_all(:,j,k)).^2) / sum((Yt_train - mean(Yt_train)).^2);
        R2_test(j,k) = 1 - sum((Yt_test - Y_pred_test_all(:,j,k)).^2) / sum((Yt_test - mean(Yt_test)).^2);
        
        % RMSE 계산
        RMSE_train(j,k) = sqrt(mean((Yt_train - Y_pred_train_all(:,j,k)).^2));
        RMSE_test(j,k) = sqrt(mean((Yt_test - Y_pred_test_all(:,j,k)).^2));
        
        % MAE 계산
        MAE_train(j,k) = mean(abs(Yt_train - Y_pred_train_all(:,j,k)));
        MAE_test(j,k) = mean(abs(Yt_test - Y_pred_test_all(:,j,k)));
    end
    
    fprintf('  %s 완료:\n', output_names{j});
    fprintf('    RBF: R²=%.3f, RMSE=%.3f, MAE=%.3f\n', R2_test(j,1), RMSE_test(j,1), MAE_test(j,1));
    fprintf('    Linear: R²=%.3f, RMSE=%.3f, MAE=%.3f\n', R2_test(j,2), RMSE_test(j,2), MAE_test(j,2));
    fprintf('    Ensemble: R²=%.3f, RMSE=%.3f, MAE=%.3f\n', R2_test(j,3), RMSE_test(j,3), MAE_test(j,3));
end
fprintf('\n모든 모델 학습 완료!\n');

%% 3. 역정규화
fprintf('\n예측값 역정규화 중...\n');
Y_pred_train_denorm = zeros(size(Y_pred_train_all));
Y_pred_test_denorm = zeros(size(Y_pred_test_all));

for k = 1:3
    for j = 1:num_outputs
        Y_pred_train_denorm(:,j,k) = Y_pred_train_all(:,j,k) * Y_std(j) + Y_mean(j);
        Y_pred_test_denorm(:,j,k)  = Y_pred_test_all(:,j,k)  * Y_std(j) + Y_mean(j);
    end
end
fprintf('역정규화 완료\n');

%% 4. 각 출력별 테스트 성능 최적 커널 선정 및 성능 지표 테이블
fprintf('\n=== 최적 커널 선정 ===\n');
[~,best_idx] = max(R2_test,[],2);
Yfit_train_best = zeros(N_train, num_outputs);
Yfit_test_best  = zeros(num_samples-N_train, num_outputs);
kernel_best = cell(num_outputs,1);

for j = 1:num_outputs
    Yfit_train_best(:,j) = Y_pred_train_denorm(:,j,best_idx(j));
    Yfit_test_best(:,j)  = Y_pred_test_denorm(:,j,best_idx(j));
    kernel_best{j} = kernel_names{best_idx(j)};
    fprintf('%s: %s 커널 선택 (R²=%.3f, RMSE=%.3f, MAE=%.3f, CV=%.3f)\n', ...
        output_names{j}, kernel_best{j}, R2_test(j,best_idx(j)), ...
        RMSE_test(j,best_idx(j)), MAE_test(j,best_idx(j)), CV_scores(j,best_idx(j)));
end

% 종합 성능 지표 테이블 생성
fprintf('\n=== 종합 성능 지표 테이블 ===\n');
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

%% 5. K-fold CV 결과 시각화
fprintf('\n시각화 생성 중...\n');
figure('Name','K-fold Cross-Validation Results','WindowStyle','docked');
for j = 1:num_outputs
    subplot(1,num_outputs,j);
    bar(CV_scores(j,:)); ylim([0 1]); grid on;
    set(gca,'XTickLabel',kernel_names); ylabel('CV R²');
    title(['Output ',output_names{j}]);
    % 최고 성능 표시
    [max_val, max_idx] = max(CV_scores(j,:));
    text(max_idx, max_val+0.02, sprintf('%.3f', max_val), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'red');
end
sgtitle('SVR Kernel별 K-fold Cross-Validation R²','FontSize',14,'FontWeight','bold');

%% 6. 성능 지표 비교 (RMSE, MAE, R²)
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

% 6-4. CV vs Test R² 비교
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

%% 7. 예측값 vs 실제값 CORRELATION PLOT (모든 출력변수, 모든 커널) - 훈련셋 + 테스트셋
% 커널별 색상 설정
kernel_colors = {'r', 'g', 'b'}; % RBF=red, Linear=green, Ensemble=blue
kernel_markers = {'o', 's', '^'}; % RBF=circle, Linear=square, Ensemble=triangle

figure('Name','Correlation Plots: 모든 커널 성능 (Y1-Y4, 훈련+테스트)','WindowStyle','docked');

for j = 1:num_outputs
    % 각 출력변수별로 하나의 subplot
    subplot(2,2,j);
    
    % 모든 커널에 대해 훈련셋 결과 플롯 (투명도 적용)
    hold on;
    for k = 1:3
        % 훈련셋 - 작고 투명하게
        scatter(Y_pred_train_denorm(:,j,k), Y_train(:,j), 60, ...
            kernel_colors{k}, kernel_markers{k}, 'MarkerFaceAlpha', 0.4, ...
            'MarkerEdgeAlpha', 0.6, ...
            'DisplayName', sprintf('%s Train (R²=%.3f)', kernel_names{k}, R2_train(j,k)));
    end
    
    % 모든 커널에 대해 테스트셋 결과 플롯 (진하게)
    for k = 1:3
        % 테스트셋 - 크고 진하게
        scatter(Y_pred_test_denorm(:,j,k), Y_test(:,j), 120, ...
            kernel_colors{k}, 'filled', kernel_markers{k}, ...
            'DisplayName', sprintf('%s Test (R²=%.3f)', kernel_names{k}, R2_test(j,k)));
    end
    
    % 1:1 기준선
    y_all = [Y_train(:,j); Y_test(:,j)];
    x_all = [Y_pred_train_denorm(:,j,:); Y_pred_test_denorm(:,j,:)];
    ymin = min(y_all); ymax = max(y_all);
    xmin = min(x_all(:)); xmax = max(x_all(:));
    lim_min = min(ymin, xmin); lim_max = max(ymax, xmax);
    plot([lim_min lim_max], [lim_min lim_max], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
    
    grid on; axis equal; box on;
    xlabel('예측값'); ylabel('실제값');
    title(sprintf('%s\n훈련+테스트 모든 커널 성능', output_names{j}));
    legend('Location', 'best', 'FontSize', 6);
    
    % 축 범위 설정
    xlim([lim_min*0.95 lim_max*1.05]);
    ylim([lim_min*0.95 lim_max*1.05]);
end

sgtitle('SVR 모든 커널 성능 비교 (Y1-Y4 훈련+테스트셋 결과)','FontSize',14,'FontWeight','bold');

%% 7-1. 훈련셋 전용 CORRELATION PLOT
figure('Name','Correlation Plots: 훈련셋 전용 (Y1-Y4, 모든 커널)','WindowStyle','docked');

for j = 1:num_outputs
    % 각 출력변수별로 하나의 subplot
    subplot(2,2,j);
    
    % 모든 커널에 대해 훈련셋 결과만 플롯
    hold on;
    for k = 1:3
        scatter(Y_pred_train_denorm(:,j,k), Y_train(:,j), 100, ...
            kernel_colors{k}, 'filled', kernel_markers{k}, ...
            'DisplayName', sprintf('%s (R²=%.3f)', kernel_names{k}, R2_train(j,k)));
    end
    
    % 1:1 기준선
    ymin = min(Y_train(:,j)); ymax = max(Y_train(:,j));
    xmin = min(Y_pred_train_denorm(:,j,:),[], 'all'); 
    xmax = max(Y_pred_train_denorm(:,j,:),[], 'all');
    lim_min = min(ymin, xmin); lim_max = max(ymax, xmax);
    plot([lim_min lim_max], [lim_min lim_max], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
    
    grid on; axis equal; box on;
    xlabel('예측값'); ylabel('실제값');
    title(sprintf('%s\n훈련셋 모든 커널 성능', output_names{j}));
    legend('Location', 'best', 'FontSize', 8);
    
    % 축 범위 설정
    xlim([lim_min*0.95 lim_max*1.05]);
    ylim([lim_min*0.95 lim_max*1.05]);
end

sgtitle('SVR 모든 커널 성능 비교 (Y1-Y4 훈련셋 결과)','FontSize',14,'FontWeight','bold');

%% 7-2. 테스트셋 전용 CORRELATION PLOT
figure('Name','Correlation Plots: 테스트셋 전용 (Y1-Y4, 모든 커널)','WindowStyle','docked');

for j = 1:num_outputs
    % 각 출력변수별로 하나의 subplot
    subplot(2,2,j);
    
    % 모든 커널에 대해 테스트셋 결과만 플롯
    hold on;
    for k = 1:3
        scatter(Y_pred_test_denorm(:,j,k), Y_test(:,j), 100, ...
            kernel_colors{k}, 'filled', kernel_markers{k}, ...
            'DisplayName', sprintf('%s (R²=%.3f)', kernel_names{k}, R2_test(j,k)));
    end
    
    % 1:1 기준선
    ymin = min(Y_test(:,j)); ymax = max(Y_test(:,j));
    xmin = min(Y_pred_test_denorm(:,j,:),[], 'all'); 
    xmax = max(Y_pred_test_denorm(:,j,:),[], 'all');
    lim_min = min(ymin, xmin); lim_max = max(ymax, xmax);
    plot([lim_min lim_max], [lim_min lim_max], 'k--', 'LineWidth', 1.5, 'DisplayName', '1:1 Line');
    
    grid on; axis equal; box on;
    xlabel('예측값'); ylabel('실제값');
    title(sprintf('%s\n테스트셋 모든 커널 성능', output_names{j}));
    legend('Location', 'best', 'FontSize', 8);
    
    % 축 범위 설정
    xlim([lim_min*0.95 lim_max*1.05]);
    ylim([lim_min*0.95 lim_max*1.05]);
end

sgtitle('SVR 모든 커널 성능 비교 (Y1-Y4 테스트셋 결과)','FontSize',14,'FontWeight','bold');

%% 8. 상세 성능 분석 - 출력변수별 모든 커널 성능 막대그래프
figure('Name','상세 성능 분석: 출력변수별 모든 커널','WindowStyle','docked');

% R² 성능 비교
subplot(2,2,1);
bar(R2_test'); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test R²');
title('Test R² 비교 (모든 커널)'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% 각 막대 위에 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, R2_test(j,k) + 0.02, sprintf('%.3f', R2_test(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% RMSE 성능 비교
subplot(2,2,2);
bar(RMSE_test'); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test RMSE');
title('Test RMSE 비교 (모든 커널)'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% 각 막대 위에 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, RMSE_test(j,k) + max(RMSE_test(:,k))*0.02, sprintf('%.3f', RMSE_test(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% MAE 성능 비교
subplot(2,2,3);
bar(MAE_test'); grid on;
set(gca,'XTickLabel',output_names); ylabel('Test MAE');
title('Test MAE 비교 (모든 커널)'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% 각 막대 위에 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, MAE_test(j,k) + max(MAE_test(:,k))*0.02, sprintf('%.3f', MAE_test(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

% CV 점수 비교
subplot(2,2,4);
bar(CV_scores'); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('CV R²');
title('Cross-Validation R² 비교'); legend(kernel_names, 'Location', 'best');
xtickangle(45);
% 각 막대 위에 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(j + (k-2)*0.27, CV_scores(j,k) + 0.02, sprintf('%.3f', CV_scores(j,k)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'FontWeight', 'bold');
    end
end

sgtitle('상세 성능 분석: Y1-Y4 모든 커널 성능 지표','FontSize',14,'FontWeight','bold');

%% 9. 히트맵으로 성능 매트릭스 시각화
figure('Name','성능 히트맵: 출력변수×커널','WindowStyle','docked');

% R² 히트맵
subplot(2,2,1);
imagesc(R2_test); colorbar; colormap(gca, 'hot');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('Test R² 히트맵'); xlabel('커널'); ylabel('출력변수');
% 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', R2_test(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% RMSE 히트맵
subplot(2,2,2);
imagesc(RMSE_test); colorbar; colormap(gca, 'cool');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('Test RMSE 히트맵'); xlabel('커널'); ylabel('출력변수');
% 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', RMSE_test(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% MAE 히트맵
subplot(2,2,3);
imagesc(MAE_test); colorbar; colormap(gca, 'winter');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('Test MAE 히트맵'); xlabel('커널'); ylabel('출력변수');
% 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', MAE_test(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% CV 점수 히트맵
subplot(2,2,4);
imagesc(CV_scores); colorbar; colormap(gca, 'spring');
set(gca,'XTick',1:3,'XTickLabel',kernel_names,'YTick',1:num_outputs,'YTickLabel',output_names);
title('CV R² 히트맵'); xlabel('커널'); ylabel('출력변수');
% 값 표시
for j = 1:num_outputs
    for k = 1:3
        text(k, j, sprintf('%.3f', CV_scores(j,k)), 'HorizontalAlignment', 'center', ...
            'Color', 'black', 'FontWeight', 'bold');
    end
end

sgtitle('성능 히트맵: 출력변수×커널 매트릭스','FontSize',14,'FontWeight','bold');

%% 8. 박스플롯 분포 비교 (훈련/테스트/예측, 커널별)
figure('Name','예측 분포 Boxplot (각 커널)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Yfit_train_best(:,j); Y_test(:,j); Yfit_test_best(:,j)];
    labels_box = [repmat({'훈련 실제'},N_train,1); repmat({'훈련 예측'},N_train,1); ...
                  repmat({'테스트 실제'},num_samples-N_train,1); repmat({'테스트 예측'},num_samples-N_train,1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('SVR 분포비교(훈련/테스트/예측)','FontSize',14,'FontWeight','bold');
