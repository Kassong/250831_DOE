clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Augment_2rbf_hyper.m RBF SVR 미실험 조건 예측 (하이퍼파라미터 최적화 + K-fold CV) ===\n');
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

%% 3. RBF SVR 하이퍼파라미터 최적화 및 K-fold 교차검증
fprintf('\n=== RBF SVR 하이퍼파라미터 최적화 및 K-fold CV 시작 ===\n');

% 하이퍼파라미터 검색 범위 설정
c_range = logspace(-2, 2, 8); % C: 0.01 ~ 100
gamma_range = logspace(-4, 1, 8); % gamma: 0.0001 ~ 10 (RBF용)
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1

% K-fold 설정
k_folds = 5; % 5-fold CV (16개 샘플이므로)
n_samples = size(X_train_norm, 1);

% CV 분할 생성
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

% 결과 저장 변수
best_params = cell(num_outputs, 1);
CV_scores = zeros(num_outputs, 1);
rmse_scores = zeros(num_outputs, 1);
mae_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    fprintf('\n출력변수 %s (%d/%d) RBF SVR 최적화 중...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % RBF 커널 최적화
    fprintf('  - RBF 커널 하이퍼파라미터 최적화...\n');
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
                
                % R² 계산
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
    
    best_params{j} = best_rbf_params;
    CV_scores(j) = best_score;
    fprintf('    최적 RBF 파라미터: C=%.4f, gamma=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_rbf_params.C, best_rbf_params.gamma, best_rbf_params.epsilon, best_score);
    
    % RMSE와 MAE 계산 (최적 파라미터로)
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
    
    rmse_scores(j) = sqrt(mean((Yt_train - cv_pred).^2));
    mae_scores(j) = mean(abs(Yt_train - cv_pred));
    
    fprintf('    CV RMSE=%.4f, MAE=%.4f\n', rmse_scores(j), mae_scores(j));
end

fprintf('\nRBF SVR 하이퍼파라미터 최적화 완료!\n');

%% 4. 최적화된 RBF SVR 모델 학습 및 예측
fprintf('\n=== 최적화된 RBF SVR 모델 학습 및 예측 시작 ===\n');
Y_predict_norm = zeros(size(X_predict,1), num_outputs);

for j = 1:num_outputs
    fprintf('출력변수 %s (%d/%d) 최적화된 RBF 예측 중...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % 최적화된 RBF SVR
    fprintf('  - 최적화된 RBF 커널 학습 및 예측...\n');
    rbf_params = best_params{j};
    mdl_rbf = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params.C, ...
        'KernelScale', 1/sqrt(2*rbf_params.gamma), ...
        'Epsilon', rbf_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j) = predict(mdl_rbf, X_predict_norm);
    
    fprintf('  %s 예측 완료\n', output_names{j});
end
fprintf('최적화된 RBF SVR 예측 완료!\n');

%% 5. 예측값 역정규화
fprintf('\n예측값 역정규화 중...\n');
Y_predict = zeros(size(Y_predict_norm));
for j = 1:num_outputs
    Y_predict(:,j) = Y_predict_norm(:,j) * Y_std(j) + Y_mean(j);
end
fprintf('역정규화 완료\n');

%% 6. 교차검증 결과 및 성능 지표 시각화
fprintf('\n=== 교차검증 결과 시각화 ===\n');

% 6-1. 성능 지표 막대그래프
figure('Name','RBF SVR K-fold CV 성능 지표','WindowStyle','docked');

subplot(1,3,1);
bar(CV_scores, 'FaceColor', [0.8 0.2 0.2]); ylim([0 1]); grid on;
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
bar(mae_scores, 'FaceColor', [0.6 0.2 0.8]); grid on;
set(gca,'XTickLabel',output_names); ylabel('MAE (normalized)');
title('K-fold CV MAE'); xtickangle(45);
for j = 1:num_outputs
    text(j, mae_scores(j)+max(mae_scores)*0.03, sprintf('%.3f', mae_scores(j)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

sgtitle('최적화된 RBF SVR K-fold 교차검증 성능 지표', 'FontSize', 14, 'FontWeight', 'bold');

% 6-2. 성능 지표 표 출력
fprintf('\n=== RBF SVR 성능 지표 요약 ===\n');
fprintf('%-12s%-15s%-15s%-15s\n', 'Output', 'CV_R²', 'CV_RMSE', 'CV_MAE');
fprintf(repmat('-', 1, 12 + 15*3));
fprintf('\n');

for j = 1:num_outputs
    fprintf('%-12s%-15.4f%-15.4f%-15.4f\n', output_names{j}, CV_scores(j), rmse_scores(j), mae_scores(j));
end

% 최적 파라미터 출력
fprintf('\n=== 최적 하이퍼파라미터 ===\n');
for j = 1:num_outputs
    rbf_params = best_params{j};
    fprintf('%s: C=%.4f, gamma=%.4f, epsilon=%.4f\n', ...
        output_names{j}, rbf_params.C, rbf_params.gamma, rbf_params.epsilon);
end

%% 7. 예측값 시각화
fprintf('\n=== 최적화된 RBF 시각화 생성 ===\n');

% 7-1. 박스플롯: RBF 예측 vs 실제
fprintf('박스플롯 생성 중...\n');
figure('Name','최적화된 RBF 예측 분포 Boxplot: 미실험(112) vs 실제(16)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j)];
    labels_box = [repmat({'실제 (16)'},size(Y_train,1),1); repmat({'RBF 예측 (112)'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('최적화된 RBF SVR: 실제(16) vs 미실험(112) 예측 분포비교','FontSize',14,'FontWeight','bold');

% 7-2. 산점도: RBF 예측조건별 실제값범위와 예측값
fprintf('산점도 및 히스토그램 생성 중...\n');
figure('Name','최적화된 RBF 예측 산점도 및 히스토그램: 조건별 예측값 분포','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j), 60, 'r', 'filled'); hold on;
    yline(min(Y_train(:,j)),'k:','최소실제'); yline(max(Y_train(:,j)),'k:','최대실제');
    grid on; ylabel(output_names{j}); legend('최적화 RBF SVR','location','best');
    title(['미실험 112조건 ',output_names{j},' 최적화 RBF 예측분포']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j),'FaceColor','r','EdgeAlpha',0.1); hold on;
    xline(min(Y_train(:,j)),'k:','최소실제'); xline(max(Y_train(:,j)),'k:','최대실제');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('최적화 RBF SVR','location','best');
    title(['미실험 112조건 ',output_names{j},' 최적화 RBF 예측 히스토그램']);
end
sgtitle('최적화된 RBF SVR 조건별 출력 예측 산점도 및 분포(16실제값 참조선)','FontSize',14,'FontWeight','bold');

% 7-3. 히트맵 시각화(최적화된 RBF)
fprintf('히트맵 생성 중...\n');
figure('Name','최적화된 RBF 예측값 히트맵','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' 최적화 RBF 예측 히트맵']);
    xlabel('X3*X4 조건 인덱스'); ylabel('X1*X2 조건 인덱스');
end
sgtitle('최적화된 RBF SVR 기반 예측값 히트맵','FontSize',14);

%% 8. 결과 저장
fprintf('\n=== 결과 저장 ===\n');

% 최적화된 RBF 예측값 저장
fprintf('최적화된 RBF SVR 예측값을 predict_rbf_hyper.csv로 저장 중...\n');
predict_table = array2table([X_predict, Y_predict], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_rbf_hyper.csv');
fprintf('저장 완료: predict_rbf_hyper.csv (%d개 예측조건)\n', size(X_predict,1));

% 최적 하이퍼파라미터 저장
fprintf('최적 하이퍼파라미터를 hyperparams_Augment_2rbf.mat로 저장 중...\n');
save('hyperparams_Augment_2rbf.mat', 'best_params', 'CV_scores', 'rmse_scores', 'mae_scores', ...
     'output_names');
fprintf('저장 완료: hyperparams_Augment_2rbf.mat\n');

fprintf('\n=== Augment_2rbf_hyper.m 실행 완료 ===\n');
fprintf('주요 개선사항:\n');
fprintf('1. RBF SVR 하이퍼파라미터 자동 최적화 (Grid Search: C, gamma, epsilon)\n');
fprintf('2. 5-fold 교차검증으로 성능 평가 향상\n');
fprintf('3. RMSE, MAE 지표 추가\n');
fprintf('4. 최적 파라미터 자동 저장 및 재사용\n');
fprintf('5. 향상된 시각화 및 상세한 성능 분석\n');