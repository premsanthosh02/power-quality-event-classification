% =========================================================
% train_classifiers.m
% Stage 4: Train & Compare 4 ML Classifiers
% Project: Power Quality Classification in Industrial Plants
% =========================================================
% Trains 4 classifiers on the 36-dim feature vectors:
%   Model 1: SVM    — Support Vector Machine (RBF kernel)
%   Model 2: RF     — Random Forest (100 trees)
%   Model 3: Boost  — Boosted Trees (AdaBoost, like XGBoost)
%   Model 4: NN     — Neural Network (Deep Learning Toolbox)
%
% Uses SNR=40dB data for training (realistic conditions)
% All 4 SNR levels used for robustness testing in Stage 5
% =========================================================

clc;
clear;
close all;

fprintf('==============================================\n');
fprintf('  PQ Classifier Training - Stage 4\n');
fprintf('==============================================\n\n');

% ---------------------------------------------------------
% SECTION 1: Load Feature Data
% ---------------------------------------------------------
fprintf('Loading feature data (SNR = 40 dB)...\n');
load('../Data/pq_features_SNR40.mat');

X = feature_matrix;
Y = labels;

fprintf('Feature matrix: %d samples x %d features\n', ...
    size(X,1), size(X,2));
fprintf('Classes: %s\n\n', strjoin(class_names, ', '));

% ---------------------------------------------------------
% SECTION 2: Train/Test Split (70% / 30%)
% ---------------------------------------------------------
rng(42);   % Fix random seed for reproducibility

cv = cvpartition(Y, 'HoldOut', 0.30);

X_train = X(cv.training, :);
Y_train = Y(cv.training);
X_test  = X(cv.test, :);
Y_test  = Y(cv.test);

% Ensure column vectors throughout
Y_train = Y_train(:);
Y_test  = Y_test(:);

fprintf('Dataset split:\n');
fprintf('  Training set : %d samples (70%%)\n', sum(cv.training));
fprintf('  Test set     : %d samples (30%%)\n\n', sum(cv.test));

% Store results for comparison
model_names = {'SVM', 'Random Forest', 'Boosted Trees', 'Neural Network'};
accuracies  = zeros(1, 4);
train_times = zeros(1, 4);

% =========================================================
% MODEL 1: SVM — Support Vector Machine
% =========================================================
fprintf('----------------------------------------\n');
fprintf('Training Model 1: SVM\n');
fprintf('----------------------------------------\n');
% SVM finds the optimal hyperplane separating classes.
% ECOC (Error-Correcting Output Codes) handles multiclass.
% RBF kernel maps features to infinite-dimensional space.

t_start = tic;

svm_template = templateSVM(...
    'KernelFunction', 'rbf', ...   % Radial Basis Function kernel
    'Standardize', true, ...       % Auto-normalise features
    'BoxConstraint', 1.0, ...      % Regularisation parameter C
    'KernelScale', 'auto');        % Auto-tune gamma

mdl_svm = fitcecoc(X_train, Y_train, ...
    'Learners', svm_template, ...
    'Coding', 'onevsone', ...      % One SVM per class pair
    'Verbose', 0);

train_times(1) = toc(t_start);

[pred_svm, scores_svm] = predict(mdl_svm, X_test);
pred_svm      = pred_svm(:);      % force column vector
accuracies(1) = mean(pred_svm == Y_test) * 100;

fprintf('  Training time : %.2f seconds\n', train_times(1));
fprintf('  Test accuracy : %.2f%%\n\n', accuracies(1));

% =========================================================
% MODEL 2: Random Forest
% =========================================================
fprintf('----------------------------------------\n');
fprintf('Training Model 2: Random Forest\n');
fprintf('----------------------------------------\n');
% 100 trees, each trained on random subset of samples
% and features. Final answer = majority vote of all trees.

t_start = tic;

mdl_rf = TreeBagger(100, X_train, Y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'MinLeafSize', 1, ...
    'NumPredictorsToSample', round(sqrt(size(X_train,2))), ...
    'Reproducible', true);

train_times(2) = toc(t_start);

pred_rf_cell  = predict(mdl_rf, X_test);
pred_rf       = categorical(pred_rf_cell);
pred_rf       = pred_rf(:);       % force column vector
accuracies(2) = mean(pred_rf == Y_test) * 100;

fprintf('  Training time : %.2f seconds\n', train_times(2));
fprintf('  Test accuracy : %.2f%%\n\n', accuracies(2));

% =========================================================
% MODEL 3: Boosted Trees (AdaBoost)
% =========================================================
fprintf('----------------------------------------\n');
fprintf('Training Model 3: Boosted Trees\n');
fprintf('----------------------------------------\n');
% Trees built sequentially — each one corrects previous errors.
% AdaBoostM2 = multiclass AdaBoost variant.
% FIX APPLIED: 'Verbose' removed (not valid in R2024a).

t_start = tic;

boost_template = templateTree(...
    'MaxNumSplits', 20, ...        % Shallow weak learners
    'Reproducible', true);

mdl_boost = fitcensemble(X_train, Y_train, ...
    'Method', 'AdaBoostM2', ...
    'NumLearningCycles', 200, ...
    'Learners', boost_template, ...
    'LearnRate', 0.1);             % Verbose removed for R2024a

train_times(3) = toc(t_start);

pred_boost    = predict(mdl_boost, X_test);
pred_boost    = pred_boost(:);    % force column vector
accuracies(3) = mean(pred_boost == Y_test) * 100;

fprintf('  Training time : %.2f seconds\n', train_times(3));
fprintf('  Test accuracy : %.2f%%\n\n', accuracies(3));

% =========================================================
% MODEL 4: Neural Network (Deep Learning Toolbox)
% =========================================================
fprintf('----------------------------------------\n');
fprintf('Training Model 4: Neural Network\n');
fprintf('----------------------------------------\n');
% Architecture: Input -> FC(128) -> BN -> ReLU -> Drop(0.3)
%                     -> FC(64)  -> BN -> ReLU -> Drop(0.2)
%                     -> FC(8)   -> Softmax -> Output
%
% FIX APPLIED: R2024a featureInputLayer format —
%   X: samples x features (do NOT transpose)
%   Y: column vector of categoricals (do NOT transpose)
%   ValidationData: {X_test, Y_test} — no transpose on Y

X_train_nn = X_train;      % 2800 x n_feats
X_test_nn  = X_test;       % 1200 x n_feats
Y_train_nn = Y_train(:);   % 2800 x 1 column vector
Y_test_nn  = Y_test(:);    % 1200 x 1 column vector

n_feats   = size(X_train, 2);   % 42 features
n_classes = numel(class_names); % 8 classes

% Define network layers
layers = [
    featureInputLayer(n_feats, 'Name', 'input')

    fullyConnectedLayer(128, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop1')

    fullyConnectedLayer(64, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.2, 'Name', 'drop2')

    fullyConnectedLayer(n_classes, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Training options
opts = trainingOptions('adam', ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 25, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_test_nn, Y_test_nn}, ...  % FIX: no Y' transpose
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10);

t_start = tic;
mdl_nn  = trainNetwork(X_train_nn, Y_train_nn, layers, opts);
train_times(4) = toc(t_start);

pred_nn       = classify(mdl_nn, X_test_nn);
pred_nn       = pred_nn(:);      % force column vector
accuracies(4) = mean(pred_nn == Y_test_nn) * 100;

fprintf('  Training time : %.2f seconds\n', train_times(4));
fprintf('  Test accuracy : %.2f%%\n\n', accuracies(4));

% ---------------------------------------------------------
% SECTION 3: Save All Trained Models
% ---------------------------------------------------------
fprintf('Saving trained models...\n');

save('../Data/trained_models.mat', ...
    'mdl_svm', 'mdl_rf', 'mdl_boost', 'mdl_nn', ...
    'X_train', 'Y_train', 'X_test',  'Y_test', ...
    'pred_svm', 'pred_rf', 'pred_boost', 'pred_nn', ...
    'accuracies', 'train_times', 'model_names', ...
    'class_names', 'cv');

fprintf('Models saved to: ../Data/trained_models.mat\n\n');

% ---------------------------------------------------------
% SECTION 4: Accuracy Comparison Bar Chart
% ---------------------------------------------------------
figure('Name', 'Model Accuracy Comparison', ...
       'NumberTitle', 'off', 'Position', [100, 100, 900, 500]);

bar_colours = [0.12 0.31 0.63;   % SVM       — navy blue
               0.11 0.45 0.18;   % RF        — forest green
               0.85 0.33 0.00;   % Boost     — orange
               0.40 0.08 0.55];  % Neural Net — purple

b = bar(accuracies, 'FaceColor', 'flat');
for k = 1:4
    b.CData(k,:) = bar_colours(k,:);
end

for k = 1:4
    text(k, accuracies(k) + 0.3, ...
        sprintf('%.2f%%', accuracies(k)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 13, 'FontWeight', 'bold');
end

xticks(1:4);
xticklabels(model_names);
ylabel('Test Accuracy (%)', 'FontSize', 12);
title('Classifier Accuracy Comparison — SNR = 40 dB', ...
      'FontSize', 14, 'FontWeight', 'bold');
ylim([min(accuracies) - 5, 103]);
grid on; grid minor;
set(gca, 'GridAlpha', 0.3);
yline(95, 'r--', '95% threshold', ...
    'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'left', ...
    'FontSize', 11);

if ~exist('../Figures', 'dir'), mkdir('../Figures'); end
saveas(gcf, '../Figures/Stage4_AccuracyComparison.png');

% ---------------------------------------------------------
% SECTION 5: Training Time Comparison
% ---------------------------------------------------------
figure('Name', 'Training Time Comparison', ...
       'NumberTitle', 'off', 'Position', [100, 100, 800, 400]);

bar_h = bar(train_times, 'FaceColor', 'flat');
for k = 1:4
    bar_h.CData(k,:) = bar_colours(k,:);
end

for k = 1:4
    text(k, train_times(k) + max(train_times) * 0.01, ...
        sprintf('%.1fs', train_times(k)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold');
end

xticks(1:4);
xticklabels(model_names);
ylabel('Training Time (seconds)', 'FontSize', 12);
title('Training Time per Classifier', ...
      'FontSize', 14, 'FontWeight', 'bold');
grid on; grid minor;
saveas(gcf, '../Figures/Stage4_TrainingTime.png');

% ---------------------------------------------------------
% SECTION 6: Confusion Matrix for Best Model
% ---------------------------------------------------------
[~, best_idx] = max(accuracies);
best_name     = model_names{best_idx};

switch best_idx
    case 1,  pred_best = pred_svm;
    case 2,  pred_best = pred_rf;
    case 3,  pred_best = pred_boost;
    case 4,  pred_best = pred_nn;
end

figure('Name', sprintf('Confusion Matrix — %s', best_name), ...
       'NumberTitle', 'off', 'Position', [100, 100, 750, 650]);

cm = confusionchart(Y_test, pred_best, ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized', ...
    'Title', sprintf('Confusion Matrix — %s (SNR=40dB)', best_name));
cm.FontSize = 11;

saveas(gcf, '../Figures/Stage4_ConfusionMatrix_Best.png');

% ---------------------------------------------------------
% SECTION 7: Print Full Summary
% ---------------------------------------------------------
fprintf('\n==============================================\n');
fprintf('  STAGE 4 COMPLETE — Results Summary\n');
fprintf('==============================================\n');
fprintf('%-20s %12s %12s\n', 'Model', 'Accuracy', 'Train Time');
fprintf('%s\n', repmat('-', 1, 46));
for k = 1:4
    fprintf('%-20s %10.2f%% %10.2fs\n', ...
        model_names{k}, accuracies(k), train_times(k));
end
fprintf('%s\n', repmat('-', 1, 46));
[best_acc, best_i] = max(accuracies);
fprintf('Best model  : %s (%.2f%%)\n', model_names{best_i}, best_acc);
fprintf('\nAll models saved to: ../Data/trained_models.mat\n');
fprintf('\nReady for Stage 5: Evaluation & Noise Robustness\n');
fprintf('==============================================\n');