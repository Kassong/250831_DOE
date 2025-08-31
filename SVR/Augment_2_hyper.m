clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Augment_2_hyper.m 미실험 조건 예측 (하이퍼파라미터 최적화 + K-fold CV) ===\n');
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

%% 3. K-fold 교차검증 및 하이퍼파라미터 최적화
fprintf('\n=== K-fold 교차검증 및 하이퍼파라미터 최적화 시작 ===\n');
kernel_names = {'RBF','Linear','Ensemble'};
k_folds = 5; % 5-fold CV (16개 샘플이므로)
n_samples = size(X_train_norm, 1);

% 하이퍼파라미터 검색 범위 설정
c_range = logspace(-2, 2, 10); % C: 0.01 ~ 100
gamma_range = logspace(-4, 1, 10); % gamma: 0.0001 ~ 10 (RBF용)
epsilon_range = logspace(-3, -1, 5); % epsilon: 0.001 ~ 0.1

% CV 분할 생성
cv_indices = crossvalind('Kfold', n_samples, k_folds);

% 결과 저장 변수
best_params = cell(num_outputs, 3); % 각 출력-커널별 최적 파라미터
CV_scores = zeros(num_outputs, 3); % 각 출력-커널별 최적 CV 점수
rmse_scores = zeros(num_outputs, 3);
mae_scores = zeros(num_outputs, 3);

for j = 1:num_outputs
    fprintf('\n출력변수 %s (%d/%d) 최적화 중...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % 1) RBF 커널 최적화
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
    
    best_params{j, 1} = best_rbf_params;
    CV_scores(j, 1) = best_score;
    fprintf('    최적 RBF 파라미터: C=%.4f, gamma=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_rbf_params.C, best_rbf_params.gamma, best_rbf_params.epsilon, best_score);
    
    % 2) Linear 커널 최적화
    fprintf('  - Linear 커널 하이퍼파라미터 최적화...\n');
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
            
            % R² 계산
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
    fprintf('    최적 Linear 파라미터: C=%.4f, epsilon=%.4f, CV R²=%.4f\n', ...
        best_linear_params.C, best_linear_params.epsilon, best_score);
    
    % 3) Ensemble 최적화 (각 구성요소별로 최적화 후 조합)
    fprintf('  - Ensemble 모델 최적화...\n');
    
    % Linear 구성요소 (위에서 구한 최적 파라미터 사용)
    best_ensemble_linear = best_linear_params;
    
    % Polynomial 구성요소 최적화
    best_score_poly = -inf;
    best_poly_params = struct();
    poly_orders = [2, 3]; % 다항식 차수
    
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
                
                % R² 계산
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
    
    % Ensemble CV 평가
    cv_predictions_ensemble = zeros(n_samples, 1);
    for fold = 1:k_folds
        train_idx = cv_indices ~= fold;
        test_idx = cv_indices == fold;
        
        % Linear 모델
        mdl_lin = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
            'KernelFunction', 'linear', 'BoxConstraint', best_ensemble_linear.C, ...
            'Epsilon', best_ensemble_linear.epsilon, 'Standardize', false);
        pred_lin = predict(mdl_lin, X_train_norm(test_idx,:));
        
        % Polynomial 모델
        mdl_poly = fitrsvm(X_train_norm(train_idx,:), Yt_train(train_idx), ...
            'KernelFunction', 'polynomial', 'BoxConstraint', best_poly_params.C, ...
            'PolynomialOrder', best_poly_params.order, ...
            'Epsilon', best_poly_params.epsilon, 'Standardize', false);
        pred_poly = predict(mdl_poly, X_train_norm(test_idx,:));
        
        cv_predictions_ensemble(test_idx) = (pred_lin + pred_poly) / 2;
    end
    
    % Ensemble R² 계산
    SS_res = sum((Yt_train - cv_predictions_ensemble).^2);
    SS_tot = sum((Yt_train - mean(Yt_train)).^2);
    ensemble_r2 = 1 - SS_res/SS_tot;
    
    best_params{j, 3} = struct('linear', best_ensemble_linear, 'poly', best_poly_params);
    CV_scores(j, 3) = ensemble_r2;
    
    fprintf('    최적 Ensemble 파라미터:\n');
    fprintf('      Linear: C=%.4f, epsilon=%.4f\n', best_ensemble_linear.C, best_ensemble_linear.epsilon);
    fprintf('      Poly: C=%.4f, epsilon=%.4f, order=%d\n', best_poly_params.C, best_poly_params.epsilon, best_poly_params.order);
    fprintf('      Ensemble CV R²=%.4f\n', ensemble_r2);
    
    % RMSE와 MAE 계산 (각 커널별)
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

fprintf('\nK-fold 교차검증 및 하이퍼파라미터 최적화 완료!\n');

%% 4. 최적화된 모델로 예측 수행
fprintf('\n=== 최적화된 SVR 모델 학습 및 예측 시작 ===\n');
Y_predict_norm = zeros(size(X_predict,1), num_outputs, 3);

for j = 1:num_outputs
    fprintf('출력변수 %s (%d/%d) 예측 중...\n', output_names{j}, j, num_outputs);
    Yt_train = Y_train_norm(:,j);
    
    % RBF (최적 파라미터 사용)
    fprintf('  - 최적화된 RBF 커널 학습 및 예측...\n');
    rbf_params = best_params{j, 1};
    mdl_rbf = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'rbf', ...
        'BoxConstraint', rbf_params.C, ...
        'KernelScale', 1/sqrt(2*rbf_params.gamma), ...
        'Epsilon', rbf_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j,1) = predict(mdl_rbf, X_predict_norm);
    
    % Linear (최적 파라미터 사용)
    fprintf('  - 최적화된 Linear 커널 학습 및 예측...\n');
    linear_params = best_params{j, 2};
    mdl_lin = fitrsvm(X_train_norm, Yt_train, 'KernelFunction', 'linear', ...
        'BoxConstraint', linear_params.C, ...
        'Epsilon', linear_params.epsilon, 'Standardize', false);
    Y_predict_norm(:,j,2) = predict(mdl_lin, X_predict_norm);
    
    % Ensemble (최적 파라미터 사용)
    fprintf('  - 최적화된 Ensemble 모델 학습 및 예측...\n');
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
    
    fprintf('  %s 예측 완료\n', output_names{j});
end
fprintf('모든 모델 예측 완료!\n');

%% 5. 예측값 역정규화
fprintf('\n예측값 역정규화 중...\n');
Y_predict = zeros(size(Y_predict_norm));
for k = 1:3
    for j = 1:num_outputs
        Y_predict(:,j,k) = Y_predict_norm(:,j,k) * Y_std(j) + Y_mean(j);
    end
end
fprintf('역정규화 완료\n');

%% 6. 교차검증 결과 및 성능 지표 시각화
fprintf('\n=== 교차검증 결과 시각화 ===\n');

% 6-1. R² 점수
figure('Name','K-fold CV 성능 지표','WindowStyle','docked');
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

sgtitle('K-fold 교차검증 성능 지표 (최적화된 하이퍼파라미터)', 'FontSize', 14, 'FontWeight', 'bold');

% 6-4. 성능 지표 표 출력
fprintf('\n=== 성능 지표 요약 ===\n');
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
fprintf('\n=== 시각화 생성 ===\n');

% 7-1. 박스플롯: 112예측(커널별) vs 16실제
fprintf('박스플롯 생성 중...\n');
figure('Name','예측 분포 Boxplot: 미실험(112) vs 실제(16) - 최적화','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    data_box = [Y_train(:,j); Y_predict(:,j,1); Y_predict(:,j,2); Y_predict(:,j,3)];
    labels_box = [repmat({'실제'},size(Y_train,1),1); repmat({'RBF'},size(Y_predict,1),1); ...
                  repmat({'Linear'},size(Y_predict,1),1); repmat({'Ensemble'},size(Y_predict,1),1)];
    boxplot(data_box, labels_box);
    title(['Output ',output_names{j}]); ylabel(output_names{j}); grid on;
end
sgtitle('실제(16) vs 미실험(112) 예측 분포비교 (최적화)','FontSize',14,'FontWeight','bold');

% 7-2. 산점도: 예측조건별 실제값범위와 예측값
fprintf('산점도 및 히스토그램 생성 중...\n');
figure('Name','예측 산점도 및 히스토그램: 조건별 예측값 분포 - 최적화','WindowStyle','docked');
for j = 1:num_outputs
    subplot(4,2,j);
    scatter(1:size(Y_predict,1), Y_predict(:,j,1), 60, 'r', 'filled'); hold on; % RBF
    scatter(1:size(Y_predict,1), Y_predict(:,j,2), 60, 'g', 'filled');
    scatter(1:size(Y_predict,1), Y_predict(:,j,3), 60, 'b', 'filled');
    yline(min(Y_train(:,j)),'k:','최소실제'); yline(max(Y_train(:,j)),'k:','최대실제');
    grid on; ylabel(output_names{j}); legend('RBF','Linear','Ensemble','location','best');
    title(['미실험 112조건 ',output_names{j},' 예측분포 (최적화)']);

    subplot(4,2,j+num_outputs);
    histogram(Y_predict(:,j,1),'FaceColor','r','EdgeAlpha',0.1); hold on;
    histogram(Y_predict(:,j,2),'FaceColor','g','EdgeAlpha',0.1);
    histogram(Y_predict(:,j,3),'FaceColor','b','EdgeAlpha',0.1);
    xline(min(Y_train(:,j)),'k:','최소실제'); xline(max(Y_train(:,j)),'k:','최대실제');
    grid on; xlabel(output_names{j}); ylabel('count'); legend('RBF','Linear','Ensemble','location','best');
    title(['미실험 112조건 ',output_names{j},' 예측 히스토그램 (최적화)']);
end
sgtitle('조건별 출력 예측 산점도 및 분포 (최적화, 16실제값 참조선)','FontSize',14,'FontWeight','bold');

% 7-3. 히트맵 시각화
fprintf('히트맵 생성 중...\n');
figure('Name','예측값 히트맵(최적화된 RBF)','WindowStyle','docked');
for j = 1:num_outputs
    subplot(2,2,j);
    Ymat = reshape(Y_predict(:,j,1), [14, 8]);
    imagesc(Ymat); colorbar;
    title(['Output ',output_names{j},' 최적화 RBF 예측 히트맵']);
    xlabel('X3*X4 조건 인덱스'); ylabel('X1*X2 조건 인덱스');
end
sgtitle('최적화된 RBF SVR 기반 예측값 히트맵','FontSize',14);

%% 8. 앙상블 예측값 CSV 저장
fprintf('\n=== 결과 저장 ===\n');
fprintf('최적화된 앙상블 모델 예측값을 predict_hyper.csv로 저장 중...\n');
% 각 모델의 앙상블 예측값만 저장 (3번째 차원이 앙상블)
ensemble_predictions = Y_predict(:,:,3);
predict_table = array2table([X_predict, ensemble_predictions], ...
    'VariableNames',[input_names, output_names]);
writetable(predict_table, 'predict_hyper.csv');
fprintf('저장 완료: predict_hyper.csv (%d개 예측조건)\n', size(X_predict,1));

% 최적 하이퍼파라미터 저장
fprintf('최적 하이퍼파라미터를 hyperparams_Augment_2.mat로 저장 중...\n');
save('hyperparams_Augment_2.mat', 'best_params', 'CV_scores', 'rmse_scores', 'mae_scores', ...
     'output_names', 'kernel_names');
fprintf('저장 완료: hyperparams_Augment_2.mat\n');

fprintf('\n=== Augment_2_hyper.m 실행 완료 ===\n');
fprintf('주요 개선사항:\n');
fprintf('1. 하이퍼파라미터 자동 최적화 (Grid Search)\n');
fprintf('2. 5-fold 교차검증으로 성능 평가 향상\n');
fprintf('3. RMSE, MAE 지표 추가\n');
fprintf('4. 최적 파라미터 자동 저장 및 재사용\n');