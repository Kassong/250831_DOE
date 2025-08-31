% Load CSV files
individual = readtable('predict_MLSSVR.csv');
single_group = readtable('predict_MLSSVR_1G.csv');
two_group = readtable('predict_MLSSVR_2G.csv');

% Extract output columns
Y_ind = individual{:,5:8};
Y_1G = single_group{:,5:8};
Y_2G = two_group{:,5:8};

% Calculate correlation matrices
fprintf('=== Correlation Analysis Results ===\n');
fprintf('\nIndividual vs Single Group:\n');
for i=1:4
    corr_ind_1g = corrcoef(Y_ind(:,i), Y_1G(:,i));
    fprintf('Y%d: %.6f\n', i, corr_ind_1g(1,2));
end

fprintf('\nIndividual vs Two Group:\n');
for i=1:4
    corr_ind_2g = corrcoef(Y_ind(:,i), Y_2G(:,i));
    fprintf('Y%d: %.6f\n', i, corr_ind_2g(1,2));
end

fprintf('\nSingle Group vs Two Group:\n');
for i=1:4
    corr_1g_2g = corrcoef(Y_1G(:,i), Y_2G(:,i));
    fprintf('Y%d: %.6f\n', i, corr_1g_2g(1,2));
end

% Calculate mean absolute differences
fprintf('\n=== Mean Absolute Differences ===\n');
fprintf('\nIndividual vs Single Group:\n');
for i=1:4
    mad_ind_1g = mean(abs(Y_ind(:,i) - Y_1G(:,i)));
    fprintf('Y%d: %.4f\n', i, mad_ind_1g);
end

fprintf('\nIndividual vs Two Group:\n');
for i=1:4
    mad_ind_2g = mean(abs(Y_ind(:,i) - Y_2G(:,i)));
    fprintf('Y%d: %.4f\n', i, mad_ind_2g);
end

% Statistical significance test
fprintf('\n=== Statistical Tests ===\n');
fprintf('\nPaired t-test (Individual vs Single Group):\n');
for i=1:4
    [h, p] = ttest(Y_ind(:,i), Y_1G(:,i));
    fprintf('Y%d: h=%d, p=%.6f\n', i, h, p);
end

fprintf('\nPaired t-test (Individual vs Two Group):\n');
for i=1:4
    [h, p] = ttest(Y_ind(:,i), Y_2G(:,i));
    fprintf('Y%d: h=%d, p=%.6f\n', i, h, p);
end