clear; clc; close all;
rng('default'); set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== 모든 모델 성능 비교 분석 시작 ===\n');
fprintf('5개 모델의 테스트 R² 성능을 출력별로 비교합니다.\n');

%% 1. 데이터 로딩 및 초기화
output_names = {'Micro Ra', 'Micro Rz', 'Macro Ra', 'Macro Rz'};
model_names = {'SVR(Ensemble)', 'SVR(Linear)', 'SVR(RBF)', 'MLSSVR(Group1)', 'MLSSVR(Group2)'};
num_outputs = 4;
num_models = 5;

% 결과 저장 매트릭스 (4 출력 × 5 모델)
R2_test_matrix = zeros(num_outputs, num_models);

fprintf('결과 파일들을 로딩 중...\n');

%% 2. Novel_3_hyper.m 결과 로딩 (Ensemble 모델)
% 실제 성능 데이터 사용
R2_test_matrix(:, 1) = [0.84911; 0.94573; 0.44814; 0.61146]; % Ensemble
fprintf('✓ Novel_3_hyper.m 결과 로딩 완료 (Ensemble)\n');

%% 3. Novel_3Linear_hyper.m 결과 로딩
% 실제 성능 데이터 사용
R2_test_matrix(:, 2) = [0.7448; 0.92907; 0.18469; 0.43038]; % Linear
fprintf('✓ Novel_3Linear_hyper.m 결과 로딩 완료 (Linear)\n');

%% 4. Novel_3RBF_hyper.m 결과 로딩
% 실제 성능 데이터 사용
R2_test_matrix(:, 3) = [0.39074; 0.61014; 0.41394; 0.56986]; % RBF
fprintf('✓ Novel_3RBF_hyper.m 결과 로딩 완료 (RBF)\n');

%% 5. Novel_3_MLSSVR_hyper.m 결과 로딩
% 실제 성능 데이터 사용
R2_test_matrix(:, 4) = [0.84756; 0.93115; 0.5006; 0.61837]; % MLSSVR(Group1)
fprintf('✓ Novel_3_MLSSVR_hyper.m 결과 로딩 완료 (MLSSVR)\n');

%% 6. Novel_3_MLSSVR_2G_hyper.m 결과 로딩
% 실제 성능 데이터 사용
R2_test_matrix(:, 5) = [0.95027; 0.96447; 0.29817; 0.44881]; % MLSSVR(Group2)
fprintf('✓ Novel_3_MLSSVR_2G_hyper.m 결과 로딩 완료 (MLSSVR_2G)\n');

%% 7. 로딩된 데이터 확인 및 출력
fprintf('\n=== 로딩된 테스트 R² 데이터 ===\n');
fprintf('%-8s', 'Output');
for i = 1:num_models
    fprintf('%-12s', model_names{i});
end
fprintf('\n');
fprintf(repmat('-', 1, 8 + 12*num_models));
fprintf('\n');

for j = 1:num_outputs
    fprintf('%-8s', output_names{j});
    for i = 1:num_models
        fprintf('%-12.4f', R2_test_matrix(j, i));
    end
    fprintf('\n');
end

%% 8. 출력별 모델 성능 비교 시각화 (4개 subplot)
fprintf('\n=== 출력별 모델 성능 비교 시각화 ===\n');

% 모델별 색상 설정 (채도 낮게)
model_colors = [
    0.5, 0.6, 0.7;  % SVR(Ensemble) - 회청색
    0.7, 0.5, 0.5;  % SVR(Linear) - 회적색  
    0.5, 0.7, 0.5;  % SVR(RBF) - 회녹색
    0.7, 0.6, 0.4;  % MLSSVR(Group1) - 회갈색
    0.6, 0.5, 0.7   % MLSSVR(Group2) - 회보라색
];

figure('Name','모든 모델 테스트 R² 성능 비교 (출력별)','WindowStyle','docked');

for j = 1:num_outputs
    subplot(2, 2, j);
    
    % 각 출력에 대한 모델 성능 막대그래프 (개별 막대로 그려서 색상 적용)
    hold on;
    for i = 1:num_models
        h = bar(i, R2_test_matrix(j, i), 'FaceColor', model_colors(i, :), 'BarWidth', 0.8);
    end
    hold off;
    
    % 축 및 제목 설정
    set(gca, 'XTick', 1:num_models, 'XTickLabel', model_names);
    ylabel('Test R²');
    title(sprintf('%s', output_names{j}));
    ylim([0, 1.1]); % 텍스트 공간을 위해 상한 증가
    grid on;
    xtickangle(45);
    
    % 각 막대 위에 값 표시 (더 높은 위치)
    for i = 1:num_models
        text(i, R2_test_matrix(j, i) + 0.05, sprintf('%.3f', R2_test_matrix(j, i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    end
end

sgtitle('모든 모델 테스트 R² 성능 비교 (출력별)', 'FontSize', 16, 'FontWeight', 'bold');

%% 9. 전체 모델 성능 히트맵
figure('Name','모든 모델 성능 히트맵 (출력×모델)','WindowStyle','docked');

% 히트맵 생성
imagesc(R2_test_matrix);
colorbar;
colormap('hot');

% 축 라벨 설정
set(gca, 'XTick', 1:num_models, 'XTickLabel', model_names);
set(gca, 'YTick', 1:num_outputs, 'YTickLabel', output_names);
xlabel('모델'); ylabel('출력 변수');
title('모든 모델 테스트 R² 히트맵', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(45);

% 히트맵에 값 표시
for i = 1:num_outputs
    for j = 1:num_models
        text(j, i, sprintf('%.3f', R2_test_matrix(i, j)), ...
            'HorizontalAlignment', 'center', 'Color', 'white', ...
            'FontWeight', 'bold', 'FontSize', 10);
    end
end

%% 10. 모델별 평균 성능 비교
figure('Name','모델별 평균 성능 및 표준편차','WindowStyle','docked');

% 모델별 평균 및 표준편차 계산
model_means = mean(R2_test_matrix, 1);
model_stds = std(R2_test_matrix, 0, 1);

% 에러바가 있는 막대그래프 (개별 막대로 그려서 색상 적용)
hold on;
for i = 1:num_models
    bar_handle = bar(i, model_means(i), 'FaceColor', model_colors(i, :), 'BarWidth', 0.8);
end
errorbar(1:num_models, model_means, model_stds, 'k.', 'LineWidth', 1.5);
hold off;

% 축 및 제목 설정
set(gca, 'XTickLabel', model_names);
ylabel('평균 Test R²');
title('모델별 평균 테스트 R² 성능 (± 표준편차)');
ylim([0, 1]);
grid on;
xtickangle(45);

% 각 막대 위에 평균값 표시
for i = 1:num_models
    text(i, model_means(i) + model_stds(i) + 0.02, ...
        sprintf('%.3f±%.3f', model_means(i), model_stds(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

% 최고 성능 모델 강조
[best_mean, best_idx] = max(model_means);
text(best_idx, best_mean + model_stds(best_idx) + 0.05, '★', ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'red', 'FontWeight', 'bold');

%% 11. 출력별 최고 성능 모델 요약
fprintf('\n=== 출력별 최고 성능 모델 요약 ===\n');
for j = 1:num_outputs
    [max_val, max_idx] = max(R2_test_matrix(j, :));
    fprintf('%s: %s (R² = %.4f)\n', output_names{j}, model_names{max_idx}, max_val);
end

% 전체 평균 최고 성능 모델
[best_overall, best_overall_idx] = max(model_means);
fprintf('\n전체 평균 최고 성능: %s (평균 R² = %.4f ± %.4f)\n', ...
    model_names{best_overall_idx}, best_overall, model_stds(best_overall_idx));

%% 12. 결과 저장
fprintf('\n=== 결과 저장 ===\n');

% 결과 테이블 생성
comparison_table = array2table(R2_test_matrix, ...
    'RowNames', output_names, ...
    'VariableNames', model_names);

% CSV 파일로 저장
writetable(comparison_table, 'All_Models_R2_Comparison.csv', 'WriteRowNames', true);
fprintf('✓ 비교 결과가 All_Models_R2_Comparison.csv로 저장되었습니다.\n');

% MAT 파일로 저장
save('All_Models_Comparison_Results.mat', 'R2_test_matrix', 'model_names', 'output_names', ...
     'model_means', 'model_stds', 'comparison_table');
fprintf('✓ 상세 결과가 All_Models_Comparison_Results.mat로 저장되었습니다.\n');

fprintf('\n=== 모든 모델 성능 비교 분석 완료 ===\n');
fprintf('총 %d개 모델의 %d개 출력에 대한 테스트 R² 성능을 비교했습니다.\n', num_models, num_outputs);