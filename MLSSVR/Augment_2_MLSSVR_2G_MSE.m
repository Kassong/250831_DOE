clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Augment_2_MLSSVR_2G_MSE.m: 2개 그룹 MLSSVR 모델 (그룹별 MSE 최적화) 예측 시작 ===\n');
fprintf('데이터 로딩 중...\n');
T = readtable('dataset.csv');
X_train = T{:,1:4}; Y_train = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
fprintf('학습 데이터: %d개 샘플, %d개 전체 출력변수\n', size(X_train,1), size(Y_train,2));

%% 0. 출력 변수 그룹화
Y_train_g1 = Y_train(:, 1:2); output_names_g1 = output_names(1:2);
Y_train_g2 = Y_train(:, 3:4); output_names_g2 = output_names(3:4);
fprintf('출력 변수를 2개 그룹으로 분할: (1,2) / (3,4)\n');

%% 1. Taguchi OA 전체조건 생성
fprintf('\nTaguchi 직교배열 전체조건 생성 중...\n');
x1_values = [250, 750, 1250, 1750];
x2_values = [20, 40, 60, 80];
x3_values = [150, 300, 450, 600];
x4_values = [4, 8];
[X1, X2, X3, X4] = ndgrid(x1_values, x2_values, x3_values, x4_values);
X_all = [X1(:), X2(:), X3(:), X4(:)];
fprintf('전체 조건: %d개\n', size(X_all,1));

% 학습 셋 제외한 예측 대상 조건 추출
is_train = ismember(X_all, X_train, 'rows');
X_predict = X_all(~is_train,:);
fprintf('학습조건 제외 후 예측대상: %d개 조건\n', size(X_predict,1));

%% 2. 데이터 정규화 (학습셋 기준)
fprintf('\n데이터 정규화 중...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_predict_norm = (X_predict - X_mean) ./ X_std;
% 그룹별 출력 정규화
[Y_train_norm_g1, Y_mean_g1, Y_std_g1] = zscore(Y_train_g1);
[Y_train_norm_g2, Y_mean_g2, Y_std_g2] = zscore(Y_train_g2);
fprintf('정규화 완료 (입력 및 그룹별 출력)\n');

%% 3. 그룹별 MSE 기반 하이퍼파라미터 최적화 함수 정의
function [best_gamma, best_lambda, best_p, best_mse] = GridMLSSVR_GroupMSE(X, Y, cv_folds)
    gamma_range = [0.01, 0.1, 1, 10];
    lambda_range = [0.01, 0.1, 1, 10];
    p_range = [0.5, 1, 2];
    
    best_mse = Inf;
    best_gamma = gamma_range(1);
    best_lambda = lambda_range(1);
    best_p = p_range(1);
    
    for gamma = gamma_range
        for lambda = lambda_range
            for p = p_range
                % Cross-validation Group MSE 계산
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
                        
                        % 그룹 MSE 계산 (그룹 내 출력에 대한 평균 MSE)
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
end

%% 4. 그룹별 LOOCV 교차검증 (그룹별 MSE 최적화)
fprintf('\n=== 그룹별 MLSSVR 모델 그룹별 MSE 기반 LOOCV 교차검증 시작 ===\n');
n_samples = size(X_train_norm, 1);
loocv_pred_g1 = zeros(n_samples, size(Y_train_g1,2));
loocv_pred_g2 = zeros(n_samples, size(Y_train_g2,2));

% --- 그룹 1 (출력 1,2) MSE 최적화 ---
fprintf('\n--- 그룹 1 (출력 1,2) 그룹 MSE 기반 검증 중 ---\n');
fprintf('그룹 1 MSE 기반 하이퍼파라미터 그리드 서치 중...\n');
[gamma_opt_g1, lambda_opt_g1, p_opt_g1, mse_opt_g1] = GridMLSSVR_GroupMSE(X_train_norm, Y_train_norm_g1, 5);
fprintf('최적 파라미터(G1, 그룹 MSE=%.6f): gamma=%.4f, lambda=%.4f, p=%.4f\n', mse_opt_g1, gamma_opt_g1, lambda_opt_g1, p_opt_g1);

loocv_mse_g1 = 0;
for i = 1:n_samples
    if mod(i,5)==0, fprintf('  G1 LOOCV %d/%d...\n', i, n_samples); end
    X_loo_train = X_train_norm([1:i-1, i+1:end], :); X_loo_test = X_train_norm(i, :);
    Y_loo_train = Y_train_norm_g1([1:i-1, i+1:end], :); Y_loo_test = Y_train_norm_g1(i, :);
    [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train, gamma_opt_g1, lambda_opt_g1, p_opt_g1);
    [pred_loo, ~, ~] = MLSSVRPredict(X_loo_test, Y_loo_test, X_loo_train, alpha_loo, b_loo, lambda_opt_g1, p_opt_g1);
    loocv_pred_g1(i, :) = pred_loo;
    loocv_mse_g1 = loocv_mse_g1 + mean((Y_loo_test - pred_loo).^2);
end

% --- 그룹 2 (출력 3,4) MSE 최적화 ---
fprintf('\n--- 그룹 2 (출력 3,4) 그룹 MSE 기반 검증 중 ---\n');
fprintf('그룹 2 MSE 기반 하이퍼파라미터 그리드 서치 중...\n');
[gamma_opt_g2, lambda_opt_g2, p_opt_g2, mse_opt_g2] = GridMLSSVR_GroupMSE(X_train_norm, Y_train_norm_g2, 5);
fprintf('최적 파라미터(G2, 그룹 MSE=%.6f): gamma=%.4f, lambda=%.4f, p=%.4f\n', mse_opt_g2, gamma_opt_g2, lambda_opt_g2, p_opt_g2);

loocv_mse_g2 = 0;
for i = 1:n_samples
    if mod(i,5)==0, fprintf('  G2 LOOCV %d/%d...\n', i, n_samples); end
    X_loo_train = X_train_norm([1:i-1, i+1:end], :); X_loo_test = X_train_norm(i, :);
    Y_loo_train = Y_train_norm_g2([1:i-1, i+1:end], :); Y_loo_test = Y_train_norm_g2(i, :);
    [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train, gamma_opt_g2, lambda_opt_g2, p_opt_g2);
    [pred_loo, ~, ~] = MLSSVRPredict(X_loo_test, Y_loo_test, X_loo_train, alpha_loo, b_loo, lambda_opt_g2, p_opt_g2);
    loocv_pred_g2(i, :) = pred_loo;
    loocv_mse_g2 = loocv_mse_g2 + mean((Y_loo_test - pred_loo).^2);
end

% LOOCV R² 계산 및 통합
LOOCV_scores_g1 = 1 - sum((Y_train_norm_g1 - loocv_pred_g1).^2) ./ sum((Y_train_norm_g1 - mean(Y_train_norm_g1)).^2);
LOOCV_scores_g2 = 1 - sum((Y_train_norm_g2 - loocv_pred_g2).^2) ./ sum((Y_train_norm_g2 - mean(Y_train_norm_g2)).^2);
LOOCV_scores = [LOOCV_scores_g1, LOOCV_scores_g2]';
MSE_scores = [loocv_mse_g1/n_samples, loocv_mse_g2/n_samples];

for j=1:length(output_names), fprintf('  %s LOOCV R² = %.4f\n', output_names{j}, LOOCV_scores(j)); end
fprintf('그룹 1 LOOCV MSE = %.6f, 그룹 2 LOOCV MSE = %.6f\n', MSE_scores(1), MSE_scores(2));
fprintf('MLSSVR 그룹별 MSE 기반 LOOCV 교차검증 완료!\n');

%% 5. 그룹별 MLSSVR 모델 학습 및 예측
fprintf('\n=== 그룹별 MLSSVR 모델 학습 및 예측 시작 ===\n');
% --- 그룹 1 (출력 1,2) ---
fprintf('그룹 1 모델 학습 및 예측 중...\n');
[alpha_g1, b_g1] = MLSSVRTrain(X_train_norm, Y_train_norm_g1, gamma_opt_g1, lambda_opt_g1, p_opt_g1);
Y_predict_dummy_g1 = zeros(size(X_predict,1), size(Y_train_g1,2));
[Y_predict_norm_g1, ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy_g1, X_train_norm, alpha_g1, b_g1, lambda_opt_g1, p_opt_g1);

% --- 그룹 2 (출력 3,4) ---
fprintf('그룹 2 모델 학습 및 예측 중...\n');
[alpha_g2, b_g2] = MLSSVRTrain(X_train_norm, Y_train_norm_g2, gamma_opt_g2, lambda_opt_g2, p_opt_g2);
Y_predict_dummy_g2 = zeros(size(X_predict,1), size(Y_train_g2,2));
[Y_predict_norm_g2, ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy_g2, X_train_norm, alpha_g2, b_g2, lambda_opt_g2, p_opt_g2);
fprintf('그룹별 예측 완료!\n');

%% 6. 예측값 역정규화 및 통합
fprintf('\n예측값 역정규화 중...\n');
Y_predict_g1 = Y_predict_norm_g1 .* Y_std_g1 + Y_mean_g1;
Y_predict_g2 = Y_predict_norm_g2 .* Y_std_g2 + Y_mean_g2;
Y_predict = [Y_predict_g1, Y_predict_g2]; % 최종 예측값 통합
fprintf('역정규화 및 결과 통합 완료\n');

%% 7. LOOCV 결과 시각화
fprintf('\n=== LOOCV 결과 시각화 ===\n');
figure('Name','2-Group MLSSVR (그룹별 MSE 최적화) LOOCV 교차검증 결과','WindowStyle','docked');
subplot(1,2,1);
bar(LOOCV_scores); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('LOOCV R²');
title('2-Group MLSSVR (그룹별 MSE 최적화) LOOCV R²'); xtickangle(45);
for j = 1:length(LOOCV_scores)
    text(j, LOOCV_scores(j)+0.05, sprintf('%.3f', LOOCV_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

subplot(1,2,2);
bar(MSE_scores); grid on;
set(gca,'XTickLabel',{'그룹1 (Y1,Y2)', '그룹2 (Y3,Y4)'}); ylabel('그룹 LOOCV MSE');
title('2-Group MLSSVR 그룹별 LOOCV MSE'); xtickangle(45);
for j = 1:length(MSE_scores)
    text(j, MSE_scores(j)+0.005, sprintf('%.4f', MSE_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% 8. 예측값 시각화 (기존과 동일)
fprintf('\n=== 시각화 생성 ===\n');
% 8-1. 박스플롯
figure('Name','MLSSVR (그룹별 MSE 최적화) 예측 분포 Boxplot: 미실험(112) vs 실제(16)','WindowStyle','docked');
for j = 1:length(output_names)
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'실제 (16)'},size(Y_train,1),1); repmat({'MLSSVR 그룹별MSE예측 (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('2-Group MLSSVR (그룹별 MSE 최적화): 실제(16) vs 미실험(112) 예측 분포비교','FontSize',14,'FontWeight','bold');

% 8-2. 산점도 및 히스토그램
figure('Name','MLSSVR (그룹별 MSE 최적화) 예측 산점도 및 히스토그램: 조건별 예측값 분포','WindowStyle','docked');
for j = 1:length(output_names)
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, [0.8 0.2 0.8], 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','최소실제'); yline(max(Y_train(:,j)),'k:','최대실제');
    grid on; ylabel(output_names{j}); legend('MLSSVR 그룹별MSE예측','location','best');
    title(['미실험 112조건 ',output_names{j},' 그룹별MSE최적화 예측분포']);
    
    subplot(4,2,j+length(output_names));
    histogram(Y_predict(:,j),'FaceColor',[0.8 0.2 0.8],'EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','최소실제'); xline(max(Y_train(:,j)),'k:','최대실제');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('MLSSVR 그룹별MSE예측','location','best');
    title(['미실험 112조건 ',output_names{j},' 그룹별MSE최적화 예측 히스토그램']);
end
sgtitle('2-Group MLSSVR (그룹별 MSE 최적화) 조건별 출력 예측 산점도 및 분포(16실제값 참조선)','FontSize',14,'FontWeight','bold');

% 8-3. 히트맵
figure('Name','MLSSVR (그룹별 MSE 최적화) 예측값 히트맵','WindowStyle','docked');
for j = 1:length(output_names)
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' 그룹별MSE최적화 예측 히트맵']);
    xlabel('X3*X4 조건 인덱스'); ylabel('X1*X2 조건 인덱스');
end
sgtitle('2-Group MLSSVR (그룹별 MSE 최적화) 기반 예측값 히트맵','FontSize',14);

%% 9. 2-Group MLSSVR 예측값 CSV 저장
fprintf('\n=== 결과 저장 ===\n');
fprintf('2-Group MLSSVR (그룹별 MSE 최적화) 모델 예측값을 predict_MLSSVR_2G_MSE.csv로 저장 중...\n');
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_MLSSVR_2G_MSE.csv');
fprintf('저장 완료: predict_MLSSVR_2G_MSE.csv (%d개 예측조건)\n', size(X_predict,1));
fprintf('\n=== Augment_2_MLSSVR_2G_MSE.m 실행 완료 ===\n');