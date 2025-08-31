clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Basic_1_MLSSVR.m MLSSVR 모델 학습 및 평가 시작 ===\n');
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

%% 2. MLSSVR 모델 학습 및 예측
fprintf('\n=== MLSSVR 모델 학습 시작 ===\n');

% MLSSVR 하이퍼파라미터 설정
fprintf('하이퍼파라미터 그리드 서치 중...\n');
[gamma_opt, lambda_opt, p_opt, MSE_opt] = GridMLSSVR(X_train_norm, Y_train_norm, 5);
fprintf('최적 파라미터: gamma=%.4f, lambda=%.4f, p=%.4f (MSE=%.6f)\n', gamma_opt, lambda_opt, p_opt, MSE_opt);

% MLSSVR 학습
fprintf('MLSSVR 모델 학습 중...\n');
[alpha, b] = MLSSVRTrain(X_train_norm, Y_train_norm, gamma_opt, lambda_opt, p_opt);

% 학습 데이터 예측
fprintf('학습 데이터 예측 중...\n');
[Y_pred_train_norm, ~, ~] = MLSSVRPredict(X_train_norm, Y_train_norm, X_train_norm, alpha, b, lambda_opt, p_opt);

% 테스트 데이터 예측
fprintf('테스트 데이터 예측 중...\n');
[Y_pred_test_norm, ~, ~] = MLSSVRPredict(X_test_norm, Y_test_norm, X_train_norm, alpha, b, lambda_opt, p_opt);

% R² 계산
R2_train = zeros(num_outputs, 1);
R2_test = zeros(num_outputs, 1);
for j = 1:num_outputs
    % 학습 R²
    SS_res_train = sum((Y_train_norm(:,j) - Y_pred_train_norm(:,j)).^2);
    SS_tot_train = sum((Y_train_norm(:,j) - mean(Y_train_norm(:,j))).^2);
    R2_train(j) = 1 - SS_res_train/SS_tot_train;
    
    % 테스트 R²
    SS_res_test = sum((Y_test_norm(:,j) - Y_pred_test_norm(:,j)).^2);
    SS_tot_test = sum((Y_test_norm(:,j) - mean(Y_test_norm(:,j))).^2);
    R2_test(j) = 1 - SS_res_test/SS_tot_test;
    
    fprintf('  %s: R²_train=%.3f, R²_test=%.3f\n', output_names{j}, R2_train(j), R2_test(j));
end
fprintf('MLSSVR 모델 학습 완료!\n');

%% 3. 역정규화
fprintf('\n예측값 역정규화 중...\n');
Y_pred_train_denorm = zeros(size(Y_pred_train_norm));
Y_pred_test_denorm = zeros(size(Y_pred_test_norm));

for j = 1:num_outputs
    Y_pred_train_denorm(:,j) = Y_pred_train_norm(:,j) * Y_std(j) + Y_mean(j);
    Y_pred_test_denorm(:,j)  = Y_pred_test_norm(:,j)  * Y_std(j) + Y_mean(j);
end
fprintf('역정규화 완료\n');

%% 4. MLSSVR 성능 출력
fprintf('\n=== MLSSVR 성능 요약 ===\n');
Yfit_train_best = Y_pred_train_denorm;
Yfit_test_best  = Y_pred_test_denorm;
for j = 1:num_outputs
    fprintf('%s: MLSSVR R²_train=%.3f, R²_test=%.3f\n', output_names{j}, R2_train(j), R2_test(j));
end

%% 5. MLSSVR R² 막대 플롯
fprintf('\n시각화 생성 중...\n');
figure('Name','MLSSVR R²','WindowStyle','docked');
subplot(1,2,1);
bar(R2_train); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R² (train)');
title('MLSSVR 학습 R²'); xtickangle(45);

subplot(1,2,2);
bar(R2_test); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R² (test)');
title('MLSSVR 테스트 R²'); xtickangle(45);
sgtitle('MLSSVR 성능','FontSize',14,'FontWeight','bold');

%% 6. 예측값 vs 실제값 CORRELATION PLOT (훈련/테스트)
figure('Name','MLSSVR Correlation Plots','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    scatter(Yfit_test_best(:,j), Y_test(:,j), 120, 'r', 'filled'); hold on; % TEST
    scatter(Yfit_train_best(:,j), Y_train(:,j), 80, 'b', 'filled','MarkerFaceAlpha',0.5); % TRAIN
    xls = min([Y_test(:,j); Yfit_test_best(:,j); Y_train(:,j); Yfit_train_best(:,j)]);
    xhs = max([Y_test(:,j); Yfit_test_best(:,j); Y_train(:,j); Yfit_train_best(:,j)]);
    plot([xls xhs],[xls xhs],'k--','LineWidth',1.5); % 1:1 기준선
    grid on; axis equal; box on;
    xlabel('MLSSVR 예측값');
    ylabel('실제값');
    title(sprintf('Output: %s\nMLSSVR R²_{train}=%.3f, R²_{test}=%.3f', ...
        output_names{j}, R2_train(j), R2_test(j)));
    legend('테스트 예측','훈련 예측','1:1 기준선','Location','best');
end
sgtitle('MLSSVR 훈련/테스트 Correlation Plot','FontSize',14,'FontWeight','bold');

%% 7. 박스플롯 분포 비교 (훈련/테스트/예측)
figure('Name','MLSSVR 예측 분포 Boxplot','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Yfit_train_best(:,j); Y_test(:,j); Yfit_test_best(:,j)];
    labels_box = [repmat({'훈련 실제'},N_train,1); repmat({'훈련 예측'},N_train,1); ...
                  repmat({'테스트 실제'},num_samples-N_train,1); repmat({'테스트 예측'},num_samples-N_train,1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('MLSSVR 분포비교(훈련/테스트/예측)','FontSize',14,'FontWeight','bold');
