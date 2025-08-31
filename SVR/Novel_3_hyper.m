clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Novel_3_hyper.m 최적화된 Ensemble SVR (하이퍼파라미터 최적화 + K-fold CV) ===\n');
fprintf('학습데이터: predict.csv, 테스트데이터: dataset.csv\n');
T_train = readtable('predict.csv');      % 112개
T_test  = readtable('dataset.csv');      % 16개

X_train = T_train{:,1:4}; Y_train = T_train{:,5:8};
X_test  = T_test{:,1:4};  Y_test  = T_test{:,5:8};
input_names = T_train.Properties.VariableNames(1:4);
output_names = T_train.Properties.VariableNames(5:8);
num_outputs = size(Y_train,2);

fprintf('데이터셋 로드 완료: 학습 %d건, 테스트 %d건\n', size(X_train,1),size(X_test,1));

% 정규화(학습셋 기준)
[X_train_norm, X_mean, X_std] = zscore(X_train);
X_test_norm = (X_test - X_mean) ./ X_std;
[Y_train_norm, Y_mean, Y_std] = zscore(Y_train);
Y_test_norm = (Y_test - Y_mean) ./ Y_std;

%% 하이퍼파라미터 최적화 및 교차검증
fprintf('\n=== 하이퍼파라미터 최적화 및 K-fold 교차검증 시작 ===\n');

% 하이퍼파라미터 검색 범위 설정
c_range = logspace(-2, 2, 8); % C: 0.01 ~ 100
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1
poly_orders = [2, 3]; % 다항식 차수

% K-fold 설정
k_folds = 5;
n_train = size(X_train_norm, 1);

% CV fold 생성
cv_indices = zeros(n_train, 1);
indices = randperm(n_train);
fold_size = floor(n_train / k_folds);
for fold = 1:k_folds
    start_idx = (fold-1) * fold_size + 1;
    if fold == k_folds
        end_idx = n_train;
    else
        end_idx = fold * fold_size;
    end
    cv_indices(indices(start_idx:end_idx)) = fold;
end

% 결과 저장 변수
best_params_linear = cell(num_outputs, 1);
best_params_poly = cell(num_outputs, 1);
CV_scores = zeros(num_outputs, 1); % Ensemble CV scores
CV_scores_linear = zeros(num_outputs, 1);
CV_scores_poly = zeros(num_outputs, 1);
rmse_scores = zeros(num_outputs, 1);
mae_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    fprintf('\n출력변수 %s (%d/%d) 하이퍼파라미터 최적화...\n', output_names{j}, j, num_outputs);
    ytr = Y_train_norm(:,j);
    
    %% 1. Linear 커널 최적화
    fprintf('  - Linear 커널 하이퍼파라미터 최적화...\n');
    best_score_linear = -inf;
    best_linear = struct();
    
    for c_idx = 1:length(c_range)
        for e_idx = 1:length(epsilon_range)
            % K-fold CV
            cv_predictions = zeros(n_train, 1);
            
            for fold = 1:k_folds
                train_idx = cv_indices ~= fold;
                test_idx = cv_indices == fold;
                
                mdl = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
                    'KernelFunction', 'linear', ...
                    'BoxConstraint', c_range(c_idx), ...
                    'Epsilon', epsilon_range(e_idx), ...
                    'Standardize', false);
                
                cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
            end
            
            % R² 계산
            SS_res = sum((ytr - cv_predictions).^2);
            SS_tot = sum((ytr - mean(ytr)).^2);
            r2_score = 1 - SS_res/SS_tot;
            
            if r2_score > best_score_linear
                best_score_linear = r2_score;
                best_linear.C = c_range(c_idx);
                best_linear.epsilon = epsilon_range(e_idx);
            end
        end
    end
    
    best_params_linear{j} = best_linear;
    CV_scores_linear(j) = best_score_linear;
    fprintf('    최적 Linear 파라미터: C=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_linear.C, best_linear.epsilon, best_score_linear);
    
    %% 2. Polynomial 커널 최적화
    fprintf('  - Polynomial 커널 하이퍼파라미터 최적화...\n');
    best_score_poly = -inf;
    best_poly = struct();
    
    for c_idx = 1:length(c_range)
        for e_idx = 1:length(epsilon_range)
            for p_idx = 1:length(poly_orders)
                % K-fold CV
                cv_predictions = zeros(n_train, 1);
                
                for fold = 1:k_folds
                    train_idx = cv_indices ~= fold;
                    test_idx = cv_indices == fold;
                    
                    mdl = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
                        'KernelFunction', 'polynomial', ...
                        'BoxConstraint', c_range(c_idx), ...
                        'PolynomialOrder', poly_orders(p_idx), ...
                        'Epsilon', epsilon_range(e_idx), ...
                        'Standardize', false);
                    
                    cv_predictions(test_idx) = predict(mdl, X_train_norm(test_idx,:));
                end
                
                % R² 계산
                SS_res = sum((ytr - cv_predictions).^2);
                SS_tot = sum((ytr - mean(ytr)).^2);
                r2_score = 1 - SS_res/SS_tot;
                
                if r2_score > best_score_poly
                    best_score_poly = r2_score;
                    best_poly.C = c_range(c_idx);
                    best_poly.epsilon = epsilon_range(e_idx);
                    best_poly.order = poly_orders(p_idx);
                end
            end
        end
    end
    
    best_params_poly{j} = best_poly;
    CV_scores_poly(j) = best_score_poly;
    fprintf('    최적 Poly 파라미터: C=%.4f, epsilon=%.4f, order=%d, CV R²=%.4f\n', ...
        best_poly.C, best_poly.epsilon, best_poly.order, best_score_poly);
    
    %% 3. Ensemble 성능 평가
    fprintf('  - Ensemble 모델 K-fold CV 평가...\n');
    cv_predictions_ensemble = zeros(n_train, 1);
    
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        % 최적화된 Linear 모델
        mdl_lin = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', best_linear.C, ...
            'Epsilon', best_linear.epsilon, ...
            'Standardize', false);
        pred_lin = predict(mdl_lin, X_train_norm(test_idx,:));
        
        % 최적화된 Polynomial 모델
        mdl_poly = fitrsvm(X_train_norm(train_idx,:), ytr(train_idx), ...
            'KernelFunction', 'polynomial', ...
            'BoxConstraint', best_poly.C, ...
            'PolynomialOrder', best_poly.order, ...
            'Epsilon', best_poly.epsilon, ...
            'Standardize', false);
        pred_poly = predict(mdl_poly, X_train_norm(test_idx,:));
        
        % Ensemble 예측
        cv_predictions_ensemble(test_idx) = (pred_lin + pred_poly) / 2;
    end
    
    % Ensemble R² 계산
    SS_res = sum((ytr - cv_predictions_ensemble).^2);
    SS_tot = sum((ytr - mean(ytr)).^2);
    ensemble_r2 = 1 - SS_res/SS_tot;
    CV_scores(j) = ensemble_r2;
    
    % RMSE와 MAE 계산
    rmse_scores(j) = sqrt(mean((ytr - cv_predictions_ensemble).^2));
    mae_scores(j) = mean(abs(ytr - cv_predictions_ensemble));
    
    fprintf('    Ensemble CV R²=%.4f, RMSE=%.4f, MAE=%.4f\n', ...
        ensemble_r2, rmse_scores(j), mae_scores(j));
end

fprintf('\n하이퍼파라미터 최적화 완료!\n');

%% 최적화된 파라미터로 최종 모델 학습 및 예측
fprintf('\n=== 최적화된 모델 학습 및 예측 ===\n');

Y_pred_train_norm = zeros(size(Y_train));
Y_pred_test_norm  = zeros(size(Y_test));
RMSE_train = zeros(num_outputs,1); RMSE_test = zeros(num_outputs,1);
MAE_train  = zeros(num_outputs,1); MAE_test  = zeros(num_outputs,1);
R2_train   = zeros(num_outputs,1); R2_test   = zeros(num_outputs,1);

for j = 1:num_outputs
    fprintf('%d/%d 최적화된 모델 학습 및 예측: %s\n',j,num_outputs,output_names{j});
    ytr = Y_train_norm(:,j);
    
    % 최적화된 파라미터로 모델 학습
    linear_params = best_params_linear{j};
    poly_params = best_params_poly{j};
    
    mdl_lin  = fitrsvm(X_train_norm, ytr, ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, ...
        'Standardize', false);
    
    mdl_poly = fitrsvm(X_train_norm, ytr, ...
        'KernelFunction', 'polynomial', ...
        'BoxConstraint', poly_params.C, ...
        'PolynomialOrder', poly_params.order, ...
        'Epsilon', poly_params.epsilon, ...
        'Standardize', false);
    
    % 예측
    Y_pred_train_norm(:,j) = (predict(mdl_lin, X_train_norm) + predict(mdl_poly, X_train_norm))/2;
    Y_pred_test_norm(:,j)  = (predict(mdl_lin, X_test_norm)  + predict(mdl_poly, X_test_norm))/2;
    
    % 평가 지표(RMSE, MAE, R2)
    RMSE_train(j) = sqrt(mean((Y_pred_train_norm(:,j)-ytr).^2));
    RMSE_test(j)  = sqrt(mean((Y_pred_test_norm(:,j)-Y_test_norm(:,j)).^2));
    MAE_train(j)  = mean(abs(Y_pred_train_norm(:,j)-ytr));
    MAE_test(j)   = mean(abs(Y_pred_test_norm(:,j)-Y_test_norm(:,j)));
    R2_train(j)   = 1 - sum((ytr-Y_pred_train_norm(:,j)).^2)/sum((ytr-mean(ytr)).^2);
    R2_test(j)    = 1 - sum((Y_test_norm(:,j)-Y_pred_test_norm(:,j)).^2)/sum((Y_test_norm(:,j)-mean(Y_test_norm(:,j))).^2);
end
fprintf('완료!\n');

%% 역정규화
Y_pred_train = Y_pred_train_norm .* Y_std + Y_mean;
Y_pred_test  = Y_pred_test_norm  .* Y_std + Y_mean;

%% Hold-out 검증 (기존 코드와 호환성을 위해 유지)
fprintf('\n=== Hold-out 방식 교차검증 ===\n');
val_ratio = 0.2;
n_val = round(n_train * val_ratio);
val_indices = randperm(n_train);
val_idx = val_indices(1:n_val);
train_cv_idx = val_indices(n_val+1:end);

X_train_cv = X_train_norm(train_cv_idx, :);
Y_train_cv = Y_train_norm(train_cv_idx, :);
X_val_cv = X_train_norm(val_idx, :);
Y_val_cv = Y_train_norm(val_idx, :);

holdout_scores = zeros(num_outputs, 1);

for j = 1:num_outputs
    linear_params = best_params_linear{j};
    poly_params = best_params_poly{j};
    
    % 최적화된 파라미터로 Hold-out 검증
    mdl_lin = fitrsvm(X_train_cv, Y_train_cv(:,j), ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, ...
        'Standardize', false);
    
    mdl_poly = fitrsvm(X_train_cv, Y_train_cv(:,j), ...
        'KernelFunction', 'polynomial', ...
        'BoxConstraint', poly_params.C, ...
        'PolynomialOrder', poly_params.order, ...
        'Epsilon', poly_params.epsilon, ...
        'Standardize', false);
    
    pred_lin = predict(mdl_lin, X_val_cv);
    pred_poly = predict(mdl_poly, X_val_cv);
    val_pred = (pred_lin + pred_poly) / 2;
    
    SS_res = sum((Y_val_cv(:,j) - val_pred).^2);
    SS_tot = sum((Y_val_cv(:,j) - mean(Y_val_cv(:,j))).^2);
    holdout_scores(j) = 1 - SS_res/SS_tot;
    fprintf('  %s Hold-out R² = %.4f\n', output_names{j}, holdout_scores(j));
end

%% 전체 지표 테이블 출력
fprintf('\n=== 성능 지표 요약 ===\n');
metrics_table = table(output_names', holdout_scores, CV_scores, rmse_scores, mae_scores, ...
    RMSE_train, RMSE_test, MAE_train, MAE_test, R2_train, R2_test, ...
    'VariableNames', {'Output','Holdout_R2','CV5_R2','CV_RMSE','CV_MAE', ...
    'RMSE_train','RMSE_test','MAE_train','MAE_test','R2_train','R2_test'});
disp(metrics_table);

% 최적 파라미터 출력
fprintf('\n=== 최적 하이퍼파라미터 ===\n');
for j = 1:num_outputs
    fprintf('%s:\n', output_names{j});
    linear_params = best_params_linear{j};
    poly_params = best_params_poly{j};
    fprintf('  Linear: C=%.4f, epsilon=%.4f (R²=%.4f)\n', ...
        linear_params.C, linear_params.epsilon, CV_scores_linear(j));
    fprintf('  Poly%d: C=%.4f, epsilon=%.4f (R²=%.4f)\n', ...
        poly_params.order, poly_params.C, poly_params.epsilon, CV_scores_poly(j));
    fprintf('  Ensemble R²=%.4f\n\n', CV_scores(j));
end

%% 시각화

% 1. 교차검증 결과 비교
fprintf('\n=== 시각화 생성 ===\n');
figure('Name','최적화된 교차검증 결과','WindowStyle','docked');

subplot(2,2,1);
bar([holdout_scores, CV_scores]); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('Hold-out vs K-fold CV (최적화)'); xtickangle(45);
legend('Hold-out', 'K-fold CV', 'Location', 'best');

subplot(2,2,2);
bar([CV_scores_linear, CV_scores_poly, CV_scores]); ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('구성요소별 K-fold CV 성능'); xtickangle(45);
legend('Linear', 'Polynomial', 'Ensemble', 'Location', 'best');

subplot(2,2,3);
bar([rmse_scores, mae_scores]); grid on;
set(gca,'XTickLabel',output_names); ylabel('Error');
title('K-fold CV 오차 지표'); xtickangle(45);
legend('RMSE', 'MAE', 'Location', 'best');

subplot(2,2,4);
x = 1:num_outputs;
width = 0.35;
bar(x - width/2, holdout_scores, width, 'FaceColor', 'b'); hold on;
bar(x + width/2, CV_scores, width, 'FaceColor', 'r');
ylim([0 1]); grid on;
set(gca,'XTickLabel',output_names); ylabel('R²');
title('검증 방법 비교'); xtickangle(45);
legend('Hold-out', 'K-fold CV', 'Location', 'best');

sgtitle('하이퍼파라미터 최적화된 교차검증 결과','FontSize',14,'FontWeight','bold');

% 2. 예측값 vs 실제값 산점도
figure('Name','최적화된 앙상블 SVR 예측 Correlation','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    scatter(Y_pred_test(:,j), Y_test(:,j), 120,'r','filled'); hold on;
    scatter(Y_pred_train(:,j), Y_train(:,j),90,'b','filled','MarkerFaceAlpha',0.6);
    xls = min([Y_pred_test(:,j); Y_test(:,j); Y_pred_train(:,j); Y_train(:,j)]);
    xhs = max([Y_pred_test(:,j); Y_test(:,j); Y_pred_train(:,j); Y_train(:,j)]);
    plot([xls xhs],[xls xhs],'k--','LineWidth',1.2);
    grid on; axis equal; box on; legend('테스트','학습','1:1 참조','Location','best');
    xlabel('예측값'); ylabel('실제값');
    title(sprintf('%s (최적화)\nRMSE_train=%.3f, test=%.3f\nMAE_train=%.3f, test=%.3f\nR²_train=%.3f, test=%.3f', ...
        output_names{j}, RMSE_train(j), RMSE_test(j), MAE_train(j), MAE_test(j), R2_train(j), R2_test(j)));
end
sgtitle('최적화된 앙상블 SVR 회귀 성능 : Correlation Plot','FontSize',14,'FontWeight','bold');

% 3. Residual Plot
figure('Name','최적화된 SVR Residual Plot','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    residual = Y_test(:,j) - Y_pred_test(:,j);
    scatter(Y_pred_test(:,j), residual, 100,'filled');
    yline(0,'k--');
    grid on; xlabel('테스트 예측값'); ylabel('잔차(실제-예측)');
    title(['Output ',output_names{j},' residual (최적화)']);
end
sgtitle('최적화된 앙상블 SVR 테스트셋 잔차 분석','FontSize',14);

% 4. 성능지표 막대그래프
figure('Name','최적화된 SVR 예측 성능지표','WindowStyle','docked');
metrics = {'RMSE','MAE','R²'};
for k = 1:3
    subplot(1,3,k);
    if k==1
        bar([RMSE_train, RMSE_test],'grouped');
        set(gca,'xticklabel',output_names);
        legend('Train','Test'); ylabel('RMSE');
        title('RMSE 비교 (최적화)');
    elseif k==2
        bar([MAE_train, MAE_test],'grouped');
        set(gca,'xticklabel',output_names);
        legend('Train','Test'); ylabel('MAE');
        title('MAE 비교 (최적화)');
    else
        bar([R2_train, R2_test],'grouped');
        set(gca,'xticklabel',output_names);
        legend('Train','Test'); ylabel('R²');
        title('R² 비교 (최적화)');
    end
    grid on; xtickangle(45);
end
sgtitle('하이퍼파라미터 최적화된 SVR 앙상블 성능지표','FontSize',14);

%% 결과 저장
fprintf('\n=== 결과 저장 ===\n');

% 최적 하이퍼파라미터 저장
fprintf('최적 하이퍼파라미터를 hyperparams_Novel_3.mat로 저장 중...\n');
save('hyperparams_Novel_3.mat', 'best_params_linear', 'best_params_poly', ...
     'CV_scores', 'CV_scores_linear', 'CV_scores_poly', ...
     'rmse_scores', 'mae_scores', 'output_names');
fprintf('저장 완료: hyperparams_Novel_3.mat\n');

% 예측 결과 저장
fprintf('최적화된 모델 예측 결과를 Novel_3_predictions_hyper.mat로 저장 중...\n');
save('Novel_3_predictions_hyper.mat', 'Y_pred_train', 'Y_pred_test', 'Y_train', 'Y_test', ...
     'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test');
fprintf('저장 완료: Novel_3_predictions_hyper.mat\n');

fprintf('\n=== Novel_3_hyper.m 실행 완료 ===\n');
fprintf('주요 개선사항:\n');
fprintf('1. Linear/Polynomial 각각 하이퍼파라미터 자동 최적화\n');
fprintf('2. K-fold 교차검증으로 성능 평가 향상\n');
fprintf('3. Ensemble 모델의 구성요소별 성능 분석\n');
fprintf('4. 최적 파라미터 자동 저장 및 재사용\n');
fprintf('5. 향상된 시각화 및 상세한 성능 분석\n');