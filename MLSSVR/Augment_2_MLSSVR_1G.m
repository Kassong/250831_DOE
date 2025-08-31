clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Augment_2_MLSSVR_1G.m: 단일 그룹 MLSSVR 모델 예측 시작 ===\n');
fprintf('데이터 로딩 중...\n');
T = readtable('dataset.csv');
X_train = T{:,1:4}; Y_train = T{:,5:8};
input_names = T.Properties.VariableNames(1:4);
output_names = T.Properties.VariableNames(5:8);
num_outputs = size(Y_train,2);
fprintf('학습 데이터: %d개 샘플, %d개 출력변수\n', size(X_train,1), num_outputs);

%% 1. Taguchi OA 전체조건 생성
fprintf('\nTaguchi 직교배열 전체조건 생성 중...\n');
x1_values = [250, 750, 1250, 1750];
x2_values = [20, 40, 60, 80];
x3_values = [150, 300, 450, 600];
x4_values = [4, 8];
[X1, X2, X3, X4] = ndgrid(x1_values, x2_values, x3_values, x4_values);
X_all = [X1(:), X2(:), X3(:), X4(:)];
fprintf('전체 조건: %d개 (%d×%d×%d×%d)\n', size(X_all,1), length(x1_values), length(x2_values), length(x3_values), length(x4_values));

% 학습 셋 제외한 예측 대상 112개 조건 추출
is_train = ismember(X_all, X_train, 'rows');
X_predict = X_all(~is_train,:);
fprintf('학습조건 제외 후 예측대상: %d개 조건\n', size(X_predict,1));

%% 2. 입력 및 출력 정규화 (학습셋 기준)
fprintf('\n데이터 정규화 중...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_predict_norm = (X_predict - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
fprintf('정규화 완료 (학습셋 기준)\n');

%% 3. LOOCV 교차검증 (단일 그룹 MLSSVR)
fprintf('\n=== 단일 MLSSVR 모델(전체 출력 그룹) LOOCV 교차검증 시작 ===\n');
n_samples = size(X_train_norm, 1);

% MLSSVR 하이퍼파라미터 그리드 서치 (전체 데이터, 다중 타겟 동시 최적화)
fprintf('하이퍼파라미터 그리드 서치 중...\n');
[gamma_opt, lambda_opt, p_opt, MSE_opt] = GridMLSSVR(X_train_norm, Y_train_norm, 5);
fprintf('최적 파라미터: gamma=%.4f, lambda=%.4f, p=%.4f (MSE=%.6f)\n', gamma_opt, lambda_opt, p_opt, MSE_opt);

% LOOCV 수행
fprintf('MLSSVR LOOCV 검증 중...\n');
loocv_pred = zeros(n_samples, num_outputs);

for i = 1:n_samples
    fprintf('  LOOCV %d/%d...\n', i, n_samples);
    
    % Leave one out
    X_loo_train = X_train_norm([1:i-1, i+1:end], :);
    Y_loo_train = Y_train_norm([1:i-1, i+1:end], :);
    X_loo_test = X_train_norm(i, :);
    Y_loo_test = Y_train_norm(i, :); % 더미용
    
    % 단일 MLSSVR 모델 학습
    [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train, gamma_opt, lambda_opt, p_opt);
    
    % 예측
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
fprintf('MLSSVR LOOCV 교차검증 완료!\n');

%% 4. 단일 그룹 MLSSVR 모델 학습 및 예측
fprintf('\n=== 단일 MLSSVR 모델(전체 출력 그룹) 학습 및 예측 시작 ===\n');

% MLSSVR 학습 (전체 학습 데이터로 단일 모델 구축)
fprintf('MLSSVR 모델 학습 중...\n');
[alpha, b] = MLSSVRTrain(X_train_norm, Y_train_norm, gamma_opt, lambda_opt, p_opt);

% 미실험 조건 예측
fprintf('미실험 조건 예측 중...\n');
Y_predict_dummy = zeros(size(X_predict,1), num_outputs); % 더미 라벨
[Y_predict_norm, ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy, X_train_norm, alpha, b, lambda_opt, p_opt);
fprintf('MLSSVR 예측 완료!\n');

%% 5. 예측값 역정규화
fprintf('\n예측값 역정규화 중...\n');
Y_predict = zeros(size(Y_predict_norm));
for j = 1:num_outputs
    Y_predict(:,j) = Y_predict_norm(:,j) * Y_std(j) + Y_mean(j);
end
fprintf('역정규화 완료\n');

%% 6. LOOCV 결과 시각화
fprintf('\n=== LOOCV 결과 시각화 ===\n');
figure('Name','단일 MLSSVR LOOCV 교차검증 결과','WindowStyle','docked');
bar(LOOCV_scores); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('LOOCV R²');
title('단일 MLSSVR 모델 LOOCV 교차검증 성능'); xtickangle(45);
for j = 1:num_outputs
    text(j, LOOCV_scores(j)+0.05, sprintf('%.3f', LOOCV_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% 7. 예측값 시각화
fprintf('\n=== 시각화 생성 ===\n');
% 7-1. 박스플롯: MLSSVR 예측 vs 실제
fprintf('박스플롯 생성 중...\n');
figure('Name','MLSSVR 예측 분포 Boxplot: 미실험(112) vs 실제(16)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'실제 (16)'},size(Y_train,1),1); repmat({'MLSSVR 예측 (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('단일 MLSSVR: 실제(16) vs 미실험(112) 예측 분포비교','FontSize',14,'FontWeight','bold');

% 7-2. 산점도: MLSSVR 예측조건별 실제값범위와 예측값
fprintf('산점도 및 히스토그램 생성 중...\n');
figure('Name','MLSSVR 예측 산점도 및 히스토그램: 조건별 예측값 분포','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'b', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','최소실제'); yline(max(Y_train(:,j)),'k:','최대실제');
    grid on; ylabel(output_names{j}); legend('MLSSVR 예측','location','best');
    title(['미실험 112조건 ',output_names{j},' MLSSVR 예측분포']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j),'FaceColor','b','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','최소실제'); xline(max(Y_train(:,j)),'k:','최대실제');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('MLSSVR 예측','location','best');
    title(['미실험 112조건 ',output_names{j},' MLSSVR 예측 히스토그램']);
end
sgtitle('단일 MLSSVR 조건별 출력 예측 산점도 및 분포(16실제값 참조선)','FontSize',14,'FontWeight','bold');

% 7-3. 히트맵 시각화(MLSSVR)
fprintf('히트맵 생성 중...\n');
figure('Name','MLSSVR 예측값 히트맵','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' MLSSVR 예측 히트맵']);
    xlabel('X3*X4 조건 인덱스'); ylabel('X1*X2 조건 인덱스');
end
sgtitle('단일 MLSSVR 기반 예측값 히트맵','FontSize',14);

%% 8. 단일 그룹 MLSSVR 예측값 CSV 저장
fprintf('\n=== 결과 저장 ===\n');
fprintf('단일 그룹 MLSSVR 모델 예측값을 predict_MLSSVR_1G.csv로 저장 중...\n');
% MLSSVR 예측값 저장
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_MLSSVR_1G.csv');
fprintf('저장 완료: predict_MLSSVR_1G.csv (%d개 예측조건)\n', size(X_predict,1));
fprintf('\n=== Augment_2_MLSSVR_1G.m 실행 완료 ===\n');