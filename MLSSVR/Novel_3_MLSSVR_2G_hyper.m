clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== 2개 그룹 MLSSVR 모델 학습/평가 시작 (확장된 하이퍼파라미터) ===\n');
fprintf('학습데이터: predict_MLSSVR_2G.csv, 테스트데이터: dataset.csv\n');
T_train = readtable('predict_MLSSVR_2G.csv');      % 112개
T_test  = readtable('dataset.csv');      % 16개

X_train = T_train{:,1:4}; Y_train = T_train{:,5:8};
X_test  = T_test{:,1:4};  Y_test  = T_test{:,5:8};

input_names = T_train.Properties.VariableNames(1:4);
output_names_original = T_train.Properties.VariableNames(5:8);
output_names = {'Micro Ra', 'Micro Rz', 'Macro Ra', 'Macro Rz'};
num_outputs = size(Y_train,2);

fprintf('데이터셋 로드 완료: 학습 %d건, 테스트 %d건\n', size(X_train,1),size(X_test,1));

% 출력 변수 그룹화
Y_train_g1 = Y_train(:, 1:2); output_names_g1 = output_names(1:2);  % 그룹 1: Y1, Y2
Y_train_g2 = Y_train(:, 3:4); output_names_g2 = output_names(3:4);  % 그룹 2: Y3, Y4
Y_test_g1 = Y_test(:, 1:2);
Y_test_g2 = Y_test(:, 3:4);

fprintf('출력 변수를 2개 그룹으로 분할: 그룹1(%s,%s) / 그룹2(%s,%s)\n', ...
    output_names_g1{1}, output_names_g1{2}, output_names_g2{1}, output_names_g2{2});

% 정규화(학습셋 기준)
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_test_norm = (X_test - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
Y_test_norm = (Y_test - Y_mean) ./ Y_std;

% 그룹별 정규화
[Y_train_norm_g1, Y_mean_g1, Y_std_g1] = zscore(Y_train_g1);
[Y_train_norm_g2, Y_mean_g2, Y_std_g2] = zscore(Y_train_g2);
Y_test_norm_g1 = (Y_test_g1 - Y_mean_g1) ./ Y_std_g1;
Y_test_norm_g2 = (Y_test_g2 - Y_mean_g2) ./ Y_std_g2;

%% 확장된 하이퍼파라미터 그리드 서치 함수 정의 (그룹별)
function [best_gamma, best_lambda, best_p, best_mse] = GridMLSSVR_Extended_Group(X, Y, cv_folds, group_name)
    % 확장된 하이퍼파라미터 범위
    gamma_range = [0.001, 0.01, 0.1, 1, 10, 100];
    lambda_range = [0.001, 0.01, 0.1, 1, 10, 100];
    p_range = [0.1, 0.5, 1, 1.5, 2, 3];
    
    best_mse = Inf;
    best_gamma = gamma_range(1);
    best_lambda = lambda_range(1);
    best_p = p_range(1);
    
    total_combinations = length(gamma_range) * length(lambda_range) * length(p_range);
    current_combo = 0;
    
    fprintf('확장된 하이퍼파라미터 탐색 시작 (%s): 총 %d개 조합\n', group_name, total_combinations);
    
    for gamma = gamma_range
        for lambda = lambda_range
            for p = p_range
                current_combo = current_combo + 1;
                if mod(current_combo, 20) == 0
                    fprintf('  %s 진행률: %d/%d (%.1f%%)\n', group_name, current_combo, total_combinations, ...
                        current_combo/total_combinations*100);
                end
                
                % Cross-validation 그룹 MSE 계산
                n = size(X, 1);
                fold_size = floor(n / cv_folds);
                cv_mse = 0;
                
                for fold = 1:cv_folds
                    test_idx = (fold-1)*fold_size + 1 : min(fold*fold_size, n);
                    train_idx = setdiff(1:n, test_idx);
                    
                    X_fold_train = X(train_idx, :);
                    Y_fold_train = Y(train_idx, :);
                    X_fold_test = X(test_idx, :);
                    Y_fold_test = Y(test_idx, :);
                    
                    try
                        [alpha, b] = MLSSVRTrain(X_fold_train, Y_fold_train, gamma, lambda, p);
                        [Y_pred, ~, ~] = MLSSVRPredict(X_fold_test, Y_fold_test, X_fold_train, alpha, b, lambda, p);
                        
                        % 그룹 내 출력에 대한 평균 MSE
                        fold_mse = mean(mean((Y_fold_test - Y_pred).^2));
                        cv_mse = cv_mse + fold_mse;
                    catch
                        cv_mse = cv_mse + Inf;
                    end
                end
                
                avg_mse = cv_mse / cv_folds;
                if avg_mse < best_mse
                    best_mse = avg_mse;
                    best_gamma = gamma;
                    best_lambda = lambda;
                    best_p = p;
                end
            end
        end
    end
    
    fprintf('확장된 하이퍼파라미터 탐색 완료 (%s)!\n', group_name);
end

% 최종 결과를 저장할 변수들 초기화
cv_scores = zeros(num_outputs, 1);
RMSE_train = zeros(num_outputs,1); RMSE_test = zeros(num_outputs,1);
MAE_train  = zeros(num_outputs,1); MAE_test  = zeros(num_outputs,1);
R2_train   = zeros(num_outputs,1); R2_test   = zeros(num_outputs,1);

%% 그룹별 MLSSVR 모델 학습 및 평가 (확장된 하이퍼파라미터)

%=== 그룹 1 (출력 1,2) 모델링 ===
fprintf('\n====== 그룹 1 (%s, %s) 모델링 시작 (확장된 하이퍼파라미터) ======\n', output_names_g1{1}, output_names_g1{2});

% --- 10-fold 교차검증 (그룹 1) ---
fprintf('그룹 1: 10-fold 교차검증 시작...\n');
k_folds = 10;
n_train = size(X_train_norm, 1);

% 확장된 하이퍼파라미터 서치 (그룹 1 데이터만 사용하여 그룹 MSE 최적화)
[gamma_opt_g1, lambda_opt_g1, p_opt_g1, mse_opt_g1] = GridMLSSVR_Extended_Group(X_train_norm, Y_train_norm_g1, 5, '그룹 1');
fprintf('그룹 1 최적 파라미터 (그룹 MSE=%.6f): gamma=%.4f, lambda=%.4f, p=%.4f\n', mse_opt_g1, gamma_opt_g1, lambda_opt_g1, p_opt_g1);

cv_indices = cvpartition(n_train, 'KFold', k_folds);
cv_pred_g1 = zeros(n_train, size(Y_train_g1,2));

for fold = 1:k_folds
    train_idx = training(cv_indices, fold);
    test_idx = test(cv_indices, fold);
    
    X_cv_train = X_train_norm(train_idx, :);
    Y_cv_train_g1 = Y_train_norm_g1(train_idx, :);
    X_cv_test = X_train_norm(test_idx, :);
    
    % 그룹 1 모델 학습
    [alpha_cv_g1, b_cv_g1] = MLSSVRTrain(X_cv_train, Y_cv_train_g1, gamma_opt_g1, lambda_opt_g1, p_opt_g1);
    
    Y_cv_test_dummy_g1 = zeros(sum(test_idx), size(Y_train_g1,2));
    [pred_cv_g1, ~, ~] = MLSSVRPredict(X_cv_test, Y_cv_test_dummy_g1, X_cv_train, alpha_cv_g1, b_cv_g1, lambda_opt_g1, p_opt_g1);
    cv_pred_g1(test_idx, :) = pred_cv_g1;
end

% 그룹 1 각 출력별 CV R² 계산
for j = 1:size(Y_train_g1,2)
    SS_res_cv = sum((Y_train_norm_g1(:,j) - cv_pred_g1(:,j)).^2);
    SS_tot_cv = sum((Y_train_norm_g1(:,j) - mean(Y_train_norm_g1(:,j))).^2);
    cv_scores(j) = 1 - SS_res_cv / SS_tot_cv;
    fprintf('  그룹 1 - %s 10-fold CV R² = %.4f\n', output_names_g1{j}, cv_scores(j));
end

% --- 그룹 1 최종 모델 학습 및 예측 ---
fprintf('그룹 1: 최종 모델 학습 및 예측...\n');
[alpha_g1, b_g1] = MLSSVRTrain(X_train_norm, Y_train_norm_g1, gamma_opt_g1, lambda_opt_g1, p_opt_g1);

% 학습 데이터 예측 (그룹 1)
Y_train_dummy_g1 = zeros(size(Y_train_norm_g1));
[Y_pred_train_norm_g1, ~, ~] = MLSSVRPredict(X_train_norm, Y_train_dummy_g1, X_train_norm, alpha_g1, b_g1, lambda_opt_g1, p_opt_g1);

% 테스트 데이터 예측 (그룹 1)
Y_test_dummy_g1 = zeros(size(Y_test_norm_g1));
[Y_pred_test_norm_g1, ~, ~] = MLSSVRPredict(X_test_norm, Y_test_dummy_g1, X_train_norm, alpha_g1, b_g1, lambda_opt_g1, p_opt_g1);

%=== 그룹 2 (출력 3,4) 모델링 ===
fprintf('\n====== 그룹 2 (%s, %s) 모델링 시작 (확장된 하이퍼파라미터) ======\n', output_names_g2{1}, output_names_g2{2});

% --- 10-fold 교차검증 (그룹 2) ---
fprintf('그룹 2: 10-fold 교차검증 시작...\n');

% 확장된 하이퍼파라미터 서치 (그룹 2 데이터만 사용하여 그룹 MSE 최적화)
[gamma_opt_g2, lambda_opt_g2, p_opt_g2, mse_opt_g2] = GridMLSSVR_Extended_Group(X_train_norm, Y_train_norm_g2, 5, '그룹 2');
fprintf('그룹 2 최적 파라미터 (그룹 MSE=%.6f): gamma=%.4f, lambda=%.4f, p=%.4f\n', mse_opt_g2, gamma_opt_g2, lambda_opt_g2, p_opt_g2);

cv_pred_g2 = zeros(n_train, size(Y_train_g2,2));

for fold = 1:k_folds
    train_idx = training(cv_indices, fold);
    test_idx = test(cv_indices, fold);
    
    X_cv_train = X_train_norm(train_idx, :);
    Y_cv_train_g2 = Y_train_norm_g2(train_idx, :);
    X_cv_test = X_train_norm(test_idx, :);
    
    % 그룹 2 모델 학습
    [alpha_cv_g2, b_cv_g2] = MLSSVRTrain(X_cv_train, Y_cv_train_g2, gamma_opt_g2, lambda_opt_g2, p_opt_g2);
    
    Y_cv_test_dummy_g2 = zeros(sum(test_idx), size(Y_train_g2,2));
    [pred_cv_g2, ~, ~] = MLSSVRPredict(X_cv_test, Y_cv_test_dummy_g2, X_cv_train, alpha_cv_g2, b_cv_g2, lambda_opt_g2, p_opt_g2);
    cv_pred_g2(test_idx, :) = pred_cv_g2;
end

% 그룹 2 각 출력별 CV R² 계산
for j = 1:size(Y_train_g2,2)
    SS_res_cv = sum((Y_train_norm_g2(:,j) - cv_pred_g2(:,j)).^2);
    SS_tot_cv = sum((Y_train_norm_g2(:,j) - mean(Y_train_norm_g2(:,j))).^2);
    cv_scores(2+j) = 1 - SS_res_cv / SS_tot_cv;
    fprintf('  그룹 2 - %s 10-fold CV R² = %.4f\n', output_names_g2{j}, cv_scores(2+j));
end

% --- 그룹 2 최종 모델 학습 및 예측 ---
fprintf('그룹 2: 최종 모델 학습 및 예측...\n');
[alpha_g2, b_g2] = MLSSVRTrain(X_train_norm, Y_train_norm_g2, gamma_opt_g2, lambda_opt_g2, p_opt_g2);

% 학습 데이터 예측 (그룹 2)
Y_train_dummy_g2 = zeros(size(Y_train_norm_g2));
[Y_pred_train_norm_g2, ~, ~] = MLSSVRPredict(X_train_norm, Y_train_dummy_g2, X_train_norm, alpha_g2, b_g2, lambda_opt_g2, p_opt_g2);

% 테스트 데이터 예측 (그룹 2)
Y_test_dummy_g2 = zeros(size(Y_test_norm_g2));
[Y_pred_test_norm_g2, ~, ~] = MLSSVRPredict(X_test_norm, Y_test_dummy_g2, X_train_norm, alpha_g2, b_g2, lambda_opt_g2, p_opt_g2);

%% 그룹별 예측 결과 통합
Y_pred_train_norm = [Y_pred_train_norm_g1, Y_pred_train_norm_g2];
Y_pred_test_norm = [Y_pred_test_norm_g1, Y_pred_test_norm_g2];

%% 성능 지표 계산 (통합된 결과로)
for j = 1:num_outputs
    RMSE_train(j) = sqrt(mean((Y_train_norm(:,j) - Y_pred_train_norm(:,j)).^2));
    RMSE_test(j)  = sqrt(mean((Y_test_norm(:,j) - Y_pred_test_norm(:,j)).^2));
    MAE_train(j)  = mean(abs(Y_train_norm(:,j) - Y_pred_train_norm(:,j)));
    MAE_test(j)   = mean(abs(Y_test_norm(:,j) - Y_pred_test_norm(:,j)));
    
    SS_res_train = sum((Y_train_norm(:,j) - Y_pred_train_norm(:,j)).^2);
    SS_tot_train = sum((Y_train_norm(:,j) - mean(Y_train_norm(:,j))).^2);
    R2_train(j) = 1 - SS_res_train/SS_tot_train;
    
    SS_res_test = sum((Y_test_norm(:,j) - Y_pred_test_norm(:,j)).^2);
    SS_tot_test = sum((Y_test_norm(:,j) - mean(Y_test_norm(:,j))).^2);
    R2_test(j) = 1 - SS_res_test/SS_tot_test;
    
    fprintf('%s: Train R²=%.3f, Test R²=%.3f\n', output_names{j}, R2_train(j), R2_test(j));
end

%% 역정규화
Y_pred_train = Y_pred_train_norm .* Y_std + Y_mean;
Y_pred_test  = Y_pred_test_norm  .* Y_std + Y_mean;

%% 전체 지표 테이블 출력
fprintf('\n====== 최종 성능 요약 (2개 그룹 모델 - 확장된 하이퍼파라미터) ======\n');
metrics_table = table(output_names', cv_scores, RMSE_train, RMSE_test, MAE_train, MAE_test, R2_train, R2_test, ...
    'VariableNames', {'Output','CV_R2','RMSE_train','RMSE_test','MAE_train','MAE_test','R2_train','R2_test'});
disp(metrics_table);

%% 10-fold CV 결과 시각화
fprintf('\n=== MLSSVR 2개 그룹 10-fold CV 결과 시각화 ===\n');
figure('Name','MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) 10-fold 교차검증 결과','WindowStyle','docked');
bar(cv_scores); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('10-fold CV R²');
title('MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) 10-fold 교차검증 성능'); xtickangle(45);
for j = 1:num_outputs
    text(j, cv_scores(j)+0.05, sprintf('%.3f', cv_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 그룹 구분선 추가
xline(2.5, 'r--', 'LineWidth', 2);
text(1.5, 0.9, '그룹 1 (Y1,Y2)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'red');
text(3.5, 0.9, '그룹 2 (Y3,Y4)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'red');

%% 1. 예측값 vs 실제값 산점도 플롯 (train/test)
figure('Name','MLSSVR Two-Group Model - Prediction Correlation','WindowStyle','docked','Position',[100,100,800,800]);
for j = 1:num_outputs
    subplot(2,2,j);
    % Test 데이터만 표시 (중간 채도 색상)
    scatter(Y_pred_test(:,j), Y_test(:,j), 60, [0.8, 0.3, 0.3], 'filled', 'MarkerFaceAlpha', 0.7); hold on;
    
    % Test 데이터의 추세선 추가
    p = polyfit(Y_pred_test(:,j), Y_test(:,j), 1);
    x_trend = linspace(min(Y_pred_test(:,j)), max(Y_pred_test(:,j)), 100);
    y_trend = polyval(p, x_trend);
    plot(x_trend, y_trend, 'Color', [0.3, 0.3, 0.6], 'LineWidth', 2);
    
    % 1:1 참조선 (범례 제외) 및 축 범위 맞추기
    xls = min([Y_pred_test(:,j); Y_test(:,j)]);
    xhs = max([Y_pred_test(:,j); Y_test(:,j)]);
    plot([xls xhs],[xls xhs],'--','Color',[0.5,0.5,0.5],'LineWidth',1);
    
    % x축과 y축 범위를 동일하게 설정
    xlim([xls xhs]);
    ylim([xls xhs]);
    
    grid on; axis square; box on;
    xlabel('Predicted Values'); ylabel('Actual Values');
    title(sprintf('%s', output_names{j}));
    
    % Test R² 값을 조금 아래로 이동
    text(0.05, 0.85, sprintf('Test R² = %.3f', R2_test(j)), ...
        'Units', 'normalized', 'FontSize', 11, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
    hold off;
end
sgtitle('MLSSVR Two-Group Model - Regression Performance: Correlation Plot','FontSize',14,'FontWeight','bold');

%% 2. Residual Plot (test)
figure('Name','MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) Residual Plot (테스트셋)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    residual = Y_test(:,j) - Y_pred_test(:,j);
    scatter(Y_pred_test(:,j), residual, 100,'filled');
    yline(0,'k--');
    grid on; xlabel('테스트 예측값'); ylabel('잔차(실제-예측)');
    
    % 그룹 정보 추가
    group_info = '';
    if j <= 2
        group_info = ' (그룹 1)';
    else
        group_info = ' (그룹 2)';
    end
    
    title(['Output ',output_names{j}, group_info,' residual']);
end
sgtitle('MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) 테스트셋 잔차 분석','FontSize',14);

%% 3. Boxplot 분포 비교 (train/test/예측)
figure('Name','MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) 예측분포 Boxplot','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_pred_train(:,j); Y_test(:,j); Y_pred_test(:,j)];
    labels_box = [repmat({'Train_실제'},size(Y_train,1),1); repmat({'Train_예측'},size(Y_pred_train,1),1); ...
                  repmat({'Test_실제'},size(Y_test,1),1); repmat({'Test_예측'},size(Y_pred_test,1),1)];
    boxplot(data_box, labels_box);
    grid on; ylabel(output_names{j});
    
    % 그룹 정보 추가
    group_info = '';
    if j <= 2
        group_info = ' (그룹 1)';
    else
        group_info = ' (그룹 2)';
    end
    
    title(['Output ',output_names{j}, group_info,' 분포비교']);
end
sgtitle('MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) train/test 분포비교','FontSize',14);

%% 4. 성능지표 막대그래프 (RMSE, MAE, R2)
figure('Name','MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) 예측 성능지표(Bars)','WindowStyle','docked');
for k = 1:3
    subplot(1,3,k);
    if k==1
        bar_data = [RMSE_train, RMSE_test];
        title_text = 'RMSE 비교';
        ylabel_text = 'RMSE';
    elseif k==2
        bar_data = [MAE_train, MAE_test];
        title_text = 'MAE 비교';
        ylabel_text = 'MAE';
    else
        bar_data = [R2_train, R2_test];
        title_text = 'R² 비교';
        ylabel_text = 'R²';
    end
    h = bar(bar_data, 'grouped');
    
    % 그룹별 색상 구분
    h(1).FaceColor = [0.2 0.6 0.8]; % Train - 파란색
    h(2).FaceColor = [0.8 0.2 0.2]; % Test - 빨간색
    
    set(gca,'xticklabel',output_names);
    legend('Train','Test','Location','best'); 
    ylabel(ylabel_text);
    title(title_text);
    grid on;
    
    % 그룹 구분선 추가
    xline(2.5, 'k--', 'Alpha', 0.5);
end
sgtitle('MLSSVR 2개 그룹 모델 (확장된 하이퍼파라미터) 성능지표 (output별)','FontSize',14);

fprintf('\n=== Novel_3_MLSSVR_2G_hyper.m 실행 완료 ===\n');