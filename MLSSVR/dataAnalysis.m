clear; clc; close all;
set(0, 'DefaultFigureWindowStyle', 'docked');

fprintf('\n=== Dataset Output Variable Correlation and Variability Analysis ===\n');

%% 1. Data Loading
T = readtable('dataset.csv');
X = T{:,1:4}; % Input variables
Y = T{:,5:8}; % Output variables

input_names = T.Properties.VariableNames(1:4);
output_names = {'Ra(Micro)', 'Rz(Micro)', 'Ra(Macro)', 'Rz(Macro)'};
n_samples = size(T, 1);
n_inputs = size(X, 2);
n_outputs = size(Y, 2);

fprintf('Dataset information:\n');
fprintf('  - Number of samples: %d\n', n_samples);
fprintf('  - Input variables: %d (%s)\n', n_inputs, strjoin(input_names, ', '));
fprintf('  - Output variables: %d (%s)\n', n_outputs, strjoin(output_names, ', '));

%% 2. Basic Statistics Calculation
fprintf('\n=== Basic Statistics for Output Variables ===\n');
output_stats = table();
for i = 1:n_outputs
    stats_struct = struct();
    stats_struct.Variable = output_names{i};
    stats_struct.Mean = mean(Y(:,i));
    stats_struct.Std = std(Y(:,i));
    stats_struct.CV = std(Y(:,i)) / mean(Y(:,i)) * 100; % Coefficient of variation (%)
    stats_struct.Min = min(Y(:,i));
    stats_struct.Max = max(Y(:,i));
    stats_struct.Range = max(Y(:,i)) - min(Y(:,i));
    
    if i == 1
        output_stats = struct2table(stats_struct);
    else
        output_stats = [output_stats; struct2table(stats_struct)];
    end
    
    fprintf('%s: Mean=%.2f, Std=%.2f, CV=%.1f%%, Range=[%.2f, %.2f]\n', ...
        output_names{i}, stats_struct.Mean, stats_struct.Std, stats_struct.CV, ...
        stats_struct.Min, stats_struct.Max);
end

%% 3. Correlation Coefficient Calculation Between Output Variables
fprintf('\n=== Correlation Coefficients Between Output Variables ===\n');
corr_matrix = corrcoef(Y);
fprintf('Correlation matrix:\n');
for i = 1:n_outputs
    fprintf('%s: ', output_names{i});
    for j = 1:n_outputs
        if i == j
            fprintf('1.000  ');
        else
            fprintf('%.3f  ', corr_matrix(i,j));
        end
    end
    fprintf('\n');
end

%% 4. Correlation Coefficient Heatmap Visualization
figure('Name','Correlation Matrix Heatmap','WindowStyle','docked');
imagesc(corr_matrix);

% 부드러운 red-blue 컬러맵 생성 (논문용 - 색상 강도 낮춤)
n_colors = 256;
% 더 부드러운 색상으로 조정 (최대 강도를 0.7로 제한)
red_part = [linspace(0.9,1,n_colors/2)', linspace(0.9,0.3,n_colors/2)', linspace(0.9,0.3,n_colors/2)'];
blue_part = [linspace(0.3,0.9,n_colors/2)', linspace(0.3,0.9,n_colors/2)', linspace(1,0.9,n_colors/2)'];
soft_redblue_cmap = [blue_part; red_part];
colormap(soft_redblue_cmap);

colorbar;
caxis([-1 1]);

% 라벨 설정
set(gca, 'XTick', 1:n_outputs, 'XTickLabel', output_names);
set(gca, 'YTick', 1:n_outputs, 'YTickLabel', output_names);
xtickangle(45);

% 상관계수 값 표시 (더 선명한 텍스트)
for i = 1:n_outputs
    for j = 1:n_outputs
        text(j, i, sprintf('%.3f', corr_matrix(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, ...
            'Color', 'black', 'FontWeight', 'bold');
    end
end

title('Pearson Correlation Coefficient Matrix', 'FontSize', 14, 'FontWeight', 'bold');
axis equal; axis tight;

%% 5. 출력 변수 분포 및 변동성 분석
figure('Name','Output Variable Distribution and Variability Analysis','WindowStyle','docked');

% 5-1. 박스플롯
subplot(2,2,1);
boxplot(Y, 'Labels', output_names);
title('Output Variable Distribution (Box Plot)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Value');
grid on;
xtickangle(45);

% 5-2. 히스토그램
subplot(2,2,2);
colors = ['b', 'r', 'g', 'm'];
for i = 1:n_outputs
    histogram(Y(:,i), 'FaceColor', colors(i), 'FaceAlpha', 0.7, 'EdgeAlpha', 0.3);
    hold on;
end
legend(output_names, 'Location', 'best');
title('Output Variable Histogram', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Value'); ylabel('Frequency');
grid on;

% 5-3. 변동계수 막대그래프
subplot(2,2,3);
cv_values = (std(Y) ./ mean(Y)) * 100;
bar(cv_values, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', output_names);
title('Coefficient of Variation (CV)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CV (%)');
grid on;
xtickangle(45);

% CV 값 표시
for i = 1:n_outputs
    text(i, cv_values(i) + max(cv_values)*0.02, sprintf('%.1f%%', cv_values(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 5-4. 정규화된 분포 비교
subplot(2,2,4);
Y_normalized = zscore(Y);
boxplot(Y_normalized, 'Labels', output_names);
title('Normalized Output Variable Distribution', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Standardized Value (Z-score)');
grid on;
xtickangle(45);

%% 6. 산점도 매트릭스 (Scatter Plot Matrix)
figure('Name','Scatter Plot Matrix of Output Variables','WindowStyle','docked');
plot_idx = 1;
for i = 1:n_outputs
    for j = 1:n_outputs
        subplot(n_outputs, n_outputs, plot_idx);
        if i == j
            % 대각선: 히스토그램
            histogram(Y(:,i), 10, 'FaceColor', colors(i), 'FaceAlpha', 0.7);
            title(sprintf('%s Distribution', output_names{i}), 'FontSize', 10);
            grid on;
        else
            % 비대각선: 산점도
            scatter(Y(:,j), Y(:,i), 60, 'filled', 'MarkerFaceAlpha', 0.7);
            
            % 선형 회귀선 추가
            p = polyfit(Y(:,j), Y(:,i), 1);
            y_fit = polyval(p, Y(:,j));
            hold on;
            plot(Y(:,j), y_fit, 'r-', 'LineWidth', 2);
            
            % 상관계수 표시
            corr_val = corr_matrix(i,j);
            text(0.05, 0.95, sprintf('r = %.3f', corr_val), ...
                'Units', 'normalized', 'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', 'white', 'EdgeColor', 'black');
            
            xlabel(output_names{j}, 'FontSize', 10);
            ylabel(output_names{i}, 'FontSize', 10);
            grid on;
        end
        plot_idx = plot_idx + 1;
    end
end
sgtitle('Scatter Plot Matrix and Distribution of Output Variables', 'FontSize', 14, 'FontWeight', 'bold');

%% 7. 주성분 분석 (PCA)
fprintf('\n=== Principal Component Analysis (PCA) ===\n');
[coeff, score, latent, tsquared, explained] = pca(Y);

fprintf('Explained variance by principal components:\n');
for i = 1:n_outputs
    fprintf('  PC%d: %.1f%%\n', i, explained(i));
end
fprintf('Cumulative explained variance (PC1+PC2): %.1f%%\n', sum(explained(1:2)));

figure('Name','Principal Component Analysis Results','WindowStyle','docked');

% 7-1. 스크리 플롯
subplot(2,2,1);
bar(explained, 'FaceColor', [0.3 0.7 0.9]);
xlabel('Principal Component'); ylabel('Explained Variance (%)');
title('Scree Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
for i = 1:n_outputs
    text(i, explained(i) + max(explained)*0.02, sprintf('%.1f%%', explained(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 7-2. 누적 설명 분산
subplot(2,2,2);
plot(1:n_outputs, cumsum(explained), 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Principal Components'); ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
ylim([0 100]);

% 7-3. PC1 vs PC2 스코어 플롯
subplot(2,2,3);
scatter(score(:,1), score(:,2), 80, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('Principal Component Score Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% 샘플 번호 표시
for i = 1:n_samples
    text(score(i,1), score(i,2), sprintf('%d', i), ...
        'FontSize', 8, 'HorizontalAlignment', 'center');
end

% 7-4. 로딩 플롯 (변수 기여도)
subplot(2,2,4);
biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', output_names);
title('PCA Biplot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

%% 8. 클러스터링 분석 (계층적 클러스터링)
fprintf('\n=== Hierarchical Clustering Analysis ===\n');
figure('Name','Hierarchical Clustering Analysis','WindowStyle','docked');

% 8-1. 출력 변수 간 거리 기반 클러스터링
Y_dist = pdist(Y', 'correlation'); % 출력 변수 간 거리 (1-상관계수)
Y_linkage = linkage(Y_dist, 'average');

subplot(1,2,1);
dendrogram(Y_linkage, 'Labels', output_names);
title('Variable Cluster Dendrogram', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Distance (1-Correlation)');
xtickangle(45);

% 8-2. 샘플 간 클러스터링
Y_sample_dist = pdist(Y, 'euclidean');
Y_sample_linkage = linkage(Y_sample_dist, 'ward');

subplot(1,2,2);
dendrogram(Y_sample_linkage, 'Labels', cellstr(num2str((1:n_samples)')));
title('Sample Cluster Dendrogram', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Euclidean Distance');
xlabel('Sample Number');

%% 9. 결과 요약 및 저장
fprintf('\n=== Analysis Results Summary ===\n');
fprintf('1. Output Variable Statistics:\n');
[~, max_cv_idx] = max(cv_values);
[~, min_cv_idx] = min(cv_values);
fprintf('   - Highest variability: %s (CV=%.1f%%)\n', output_names{max_cv_idx}, cv_values(max_cv_idx));
fprintf('   - Lowest variability: %s (CV=%.1f%%)\n', output_names{min_cv_idx}, cv_values(min_cv_idx));

fprintf('2. Correlation Analysis:\n');
[max_corr, max_idx] = max(abs(corr_matrix(triu(ones(n_outputs),1) == 1)));
[row, col] = find(abs(corr_matrix) == max_corr & triu(ones(n_outputs),1) == 1);
if ~isempty(row)
    fprintf('   - Strongest correlation: %s vs %s (r=%.3f)\n', ...
        output_names{row(1)}, output_names{col(1)}, corr_matrix(row(1), col(1)));
end

[min_corr, min_idx] = min(abs(corr_matrix(triu(ones(n_outputs),1) == 1)));
[row, col] = find(abs(corr_matrix) == min_corr & triu(ones(n_outputs),1) == 1);
if ~isempty(row)
    fprintf('   - Weakest correlation: %s vs %s (r=%.3f)\n', ...
        output_names{row(1)}, output_names{col(1)}, corr_matrix(row(1), col(1)));
end

fprintf('3. Principal Component Analysis:\n');
fprintf('   - First 2 PCs explain %.1f%% of variance\n', sum(explained(1:2)));
fprintf('   - Recommended number of PCs for dimensionality reduction: %d (90%% criterion)\n', ...
    find(cumsum(explained) >= 90, 1));

% 분석 결과를 구조체로 저장
analysis_results = struct();
analysis_results.basic_stats = output_stats;
analysis_results.correlation_matrix = corr_matrix;
analysis_results.pca_explained = explained;
analysis_results.pca_coefficients = coeff;
analysis_results.cv_values = cv_values;

save('dataAnalysis_results.mat', 'analysis_results', 'Y', 'output_names');
fprintf('\nAnalysis results saved to dataAnalysis_results.mat\n');

fprintf('\n=== dataAnalysis.m execution completed ===\n');