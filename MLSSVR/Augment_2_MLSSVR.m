clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('MLSSVR-master');

fprintf('\n=== Augment_2_MLSSVR.m: "개별" MLSSVR 모델 예측 시작 ===\n');
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
is_train = ismember(X_all, X_train, 'rows');
X_predict = X_all(~is_train,:);
fprintf('예측대상: %d개 조건\n', size(X_predict,1));

%% 2. 입력 및 출력 정규화 (학습셋 기준)
fprintf('\n데이터 정규화 중...\n');
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_predict_norm = (X_predict - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
fprintf('정규화 완료\n');

%% 3. 각 출력별 개별 모델링 루프
Y_predict_norm = zeros(size(X_predict,1), num_outputs);
LOOCV_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    fprintf('\n====== 출력 변수 %s (%d/%d) 개별 모델링 시작 ======\n', output_names{j}, j, num_outputs);
    
    % 현재 출력 변수에 해당하는 Y 데이터만 선택
    Y_train_norm_single = Y_train_norm(:, j);

    % 3-1. 하이퍼파라미터 탐색 (개별 Y 데이터 사용)
    fprintf('하이퍼파라미터 그리드 서치 중...\n');
    [gamma_opt, lambda_opt, p_opt, ~] = GridMLSSVR(X_train_norm, Y_train_norm_single, 5);
    fprintf('최적 파라미터: gamma=%.4f, lambda=%.4f, p=%.4f\n', gamma_opt, lambda_opt, p_opt);

    % 3-2. LOOCV 교차검증 (개별 Y 데이터 사용)
    fprintf('LOOCV 검증 중...\n');
    n_samples = size(X_train_norm, 1);
    loocv_pred = zeros(n_samples, 1);
    for i = 1:n_samples
        X_loo_train = X_train_norm([1:i-1, i+1:end], :);
        Y_loo_train_single = Y_train_norm_single([1:i-1, i+1:end], :);
        X_loo_test = X_train_norm(i, :);
        Y_loo_test_dummy = 0; % 1x1 더미
        
        [alpha_loo, b_loo] = MLSSVRTrain(X_loo_train, Y_loo_train_single, gamma_opt, lambda_opt, p_opt);
        [pred_loo, ~, ~] = MLSSVRPredict(X_loo_test, Y_loo_test_dummy, X_loo_train, alpha_loo, b_loo, lambda_opt, p_opt);
        loocv_pred(i) = pred_loo;
    end
    SS_res = sum((Y_train_norm_single - loocv_pred).^2);
    SS_tot = sum((Y_train_norm_single - mean(Y_train_norm_single)).^2);
    LOOCV_scores(j) = 1 - SS_res/SS_tot;
    fprintf('  %s LOOCV R² = %.4f\n', output_names{j}, LOOCV_scores(j));

    % 3-3. 최종 모델 학습 및 예측 (개별 Y 데이터 사용)
    fprintf('최종 모델 학습 및 예측 중...\n');
    [alpha, b] = MLSSVRTrain(X_train_norm, Y_train_norm_single, gamma_opt, lambda_opt, p_opt);
    
    Y_predict_dummy = zeros(size(X_predict,1), 1);
    [Y_predict_norm(:, j), ~, ~] = MLSSVRPredict(X_predict_norm, Y_predict_dummy, X_train_norm, alpha, b, lambda_opt, p_opt);
end

%% 4. 역정규화 및 결과 저장
fprintf('\n\n예측값 역정규화 중...\n');
Y_predict = Y_predict_norm .* Y_std + Y_mean;
fprintf('역정규화 완료\n');

fprintf('\n결과 저장 중...\n');
predict_table = array2table([X_predict, Y_predict], 'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_MLSSVR.csv');
fprintf('저장 완료: predict_MLSSVR.csv\n');

%% 5. 시각화
fprintf('\n=== 시각화 생성 ===\n');

% 5-1. LOOCV 결과 시각화
figure('Name','개별 MLSSVR LOOCV 교차검증 결과','WindowStyle','docked');
bar(LOOCV_scores); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('LOOCV R²');
title('개별 MLSSVR 모델 LOOCV 교차검증 성능'); xtickangle(45);
for j = 1:num_outputs
    text(j, LOOCV_scores(j)+0.05, sprintf('%.3f', LOOCV_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 5-2. 박스플롯: 예측 vs 실제
figure('Name','개별 MLSSVR 예측 분포 Boxplot: 미실험(112) vs 실제(16)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'실제 (16)'},size(Y_train,1),1); repmat({'MLSSVR 예측 (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('개별 MLSSVR: 실제(16) vs 미실험(112) 예측 분포비교','FontSize',14,'FontWeight','bold');

% 5-3. 산점도 및 히스토그램: 조건별 예측값
figure('Name','개별 MLSSVR 예측 산점도 및 히스토그램','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'b', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','최소실제'); yline(max(Y_train(:,j)),'k:','최대실제');
    grid on; ylabel(output_names{j}); legend('MLSSVR 예측','location','best');
    title(['미실험 112조건 ',output_names{j},' 예측분포']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j),'FaceColor','b','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','최소실제'); xline(max(Y_train(:,j)),'k:','최대실제');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('MLSSVR 예측','location','best');
    title(['미실험 112조건 ',output_names{j},' 예측 히스토그램']);
end
sgtitle('개별 MLSSVR 조건별 출력 예측 산점도 및 분포','FontSize',14,'FontWeight','bold');

% 5-4. 히트맵 시각화
figure('Name','개별 MLSSVR 예측값 히트맵','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' 예측 히트맵']);
    xlabel('X3*X4 조건 인덱스'); ylabel('X1*X2 조건 인덱스');
end
sgtitle('개별 MLSSVR 기반 예측값 히트맵','FontSize',14);

fprintf('\n=== Augment_2_MLSSVR.m 실행 완료 ===\n');