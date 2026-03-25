% =========================================================
% evaluate_models.m
% Stage 5: Full Evaluation & Noise Robustness Analysis
% Project: Power Quality Classification in Industrial Plants
% =========================================================

clc;
clear;
close all;

fprintf('==============================================\n');
fprintf('  PQ Model Evaluation - Stage 5\n');
fprintf('==============================================\n\n');

% ---------------------------------------------------------
% SECTION 1: Load Trained Models & Data
% ---------------------------------------------------------
fprintf('Loading trained models...\n');
load('../Data/trained_models.mat');

fprintf('Loading all SNR feature datasets...\n');
snr_levels = [50, 40, 30, 20];
feat_data  = cell(1, 4);
for s = 1:4
    feat_data{s} = load(sprintf('../Data/pq_features_SNR%d.mat', ...
                                 snr_levels(s)));
end
fprintf('All data loaded.\n\n');

if ~exist('../Figures','dir'), mkdir('../Figures'); end

% Ensure all predictions are column vectors
pred_svm   = pred_svm(:);
pred_rf    = pred_rf(:);
pred_boost = pred_boost(:);
pred_nn    = pred_nn(:);
Y_test     = Y_test(:);

all_preds = {pred_svm, pred_rf, pred_boost, pred_nn};

bar_colours = [0.12 0.31 0.63;   % SVM        — navy blue
               0.11 0.45 0.18;   % RF         — forest green
               0.85 0.33 0.00;   % Boost      — orange
               0.40 0.08 0.55];  % Neural Net — purple

% =========================================================
% SECTION 2: Confusion Matrices — All 4 Models
% =========================================================
fprintf('--- Generating confusion matrices ---\n');

figure('Name','Confusion Matrices — All Models', ...
       'NumberTitle','off','Position',[50,50,1400,900]);

for m = 1:4
    subplot(2, 2, m);
    cm = confusionchart(Y_test, all_preds{m}, ...
        'RowSummary',    'row-normalized', ...
        'ColumnSummary', 'column-normalized', ...
        'Title', sprintf('%s — %.2f%%', ...
            model_names{m}, accuracies(m)));
    cm.FontSize = 8;
end

sgtitle('Confusion Matrices — All 4 Classifiers (SNR=40dB)', ...
        'FontSize',14,'FontWeight','bold');

saveas(gcf,'../Figures/Stage5_ConfusionMatrices_All.png');
fprintf('  Saved: Stage5_ConfusionMatrices_All.png\n');

% =========================================================
% SECTION 3: Per-Class F1 Score Analysis
% =========================================================
fprintf('\n--- Computing per-class F1 scores ---\n');

n_classes  = numel(class_names);
precision  = zeros(4, n_classes);
recall_arr = zeros(4, n_classes);
f1_scores  = zeros(4, n_classes);

for m = 1:4
    pred  = all_preds{m}(:);
    truth = Y_test(:);

    for c = 1:n_classes
        cls = categorical(class_names(c));

        TP = sum(pred == cls & truth == cls);
        FP = sum(pred == cls & truth ~= cls);
        FN = sum(pred ~= cls & truth == cls);

        if (TP + FP) > 0
            precision(m,c) = TP / (TP + FP);
        end
        if (TP + FN) > 0
            recall_arr(m,c) = TP / (TP + FN);
        end
        if (precision(m,c) + recall_arr(m,c)) > 0
            f1_scores(m,c) = 2 * precision(m,c) * recall_arr(m,c) / ...
                            (precision(m,c) + recall_arr(m,c));
        end
    end
end

% Print F1 table
fprintf('\nF1-Score Table (per class, per model):\n');
fprintf('%-15s', 'Class');
for m = 1:4
    fprintf('%15s', model_names{m});
end
fprintf('\n%s\n', repmat('-', 1, 15+15*4));
for c = 1:n_classes
    fprintf('%-15s', class_names{c});
    for m = 1:4
        fprintf('%14.4f ', f1_scores(m,c));
    end
    fprintf('\n');
end
fprintf('%s\n', repmat('-', 1, 15+15*4));
fprintf('%-15s', 'MEAN F1');
for m = 1:4
    fprintf('%14.4f ', mean(f1_scores(m,:)));
end
fprintf('\n');

% F1 bar chart
figure('Name','F1 Score per Class','NumberTitle','off', ...
       'Position',[50,50,1300,500]);

b = bar(f1_scores' * 100, 'grouped');
for m = 1:4
    b(m).FaceColor = bar_colours(m,:);
end

xticks(1:n_classes);
xticklabels(class_names);
xtickangle(30);
ylabel('F1 Score (%)', 'FontSize', 12);
title('Per-Class F1 Score — All 4 Models (SNR=40dB)', ...
      'FontSize',14,'FontWeight','bold');
legend(model_names,'Location','southeast','FontSize',10);
ylim([70, 102]);
grid on; grid minor;
yline(95,'k--','LineWidth',1.2);

saveas(gcf,'../Figures/Stage5_F1Scores.png');
fprintf('\n  Saved: Stage5_F1Scores.png\n');

% =========================================================
% SECTION 4: Noise Robustness Analysis
% =========================================================
fprintf('\n--- Running noise robustness analysis ---\n');

robustness = zeros(4, 4);   % 4 models x 4 SNR levels

for s = 1:4
    snr_db = snr_levels(s);
    fprintf('  Testing at SNR = %d dB...\n', snr_db);

    X_snr      = feat_data{s}.feature_matrix;
    Y_snr      = feat_data{s}.labels;
    X_snr_test = X_snr(cv.test, :);
    Y_snr_test = Y_snr(cv.test);
    Y_snr_test = Y_snr_test(:);

    % SVM
    p = predict(mdl_svm, X_snr_test);
    robustness(1,s) = mean(p(:) == Y_snr_test) * 100;

    % Random Forest
    p = categorical(predict(mdl_rf, X_snr_test));
    robustness(2,s) = mean(p(:) == Y_snr_test) * 100;

    % Boosted Trees
    p = predict(mdl_boost, X_snr_test);
    robustness(3,s) = mean(p(:) == Y_snr_test) * 100;

    % Neural Network
    p = classify(mdl_nn, X_snr_test);
    robustness(4,s) = mean(p(:) == Y_snr_test) * 100;
end

% Print robustness table
fprintf('\nNoise Robustness (Accuracy %% at each SNR level):\n');
fprintf('%-20s %8s %8s %8s %8s %8s\n', ...
    'Model','50 dB','40 dB','30 dB','20 dB','Drop');
fprintf('%s\n', repmat('-',1,60));
for m = 1:4
    drop = robustness(m,1) - robustness(m,4);
    fprintf('%-20s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n', ...
        model_names{m}, robustness(m,1), robustness(m,2), ...
        robustness(m,3), robustness(m,4), drop);
end

% Robustness line chart
figure('Name','Noise Robustness','NumberTitle','off', ...
       'Position',[50,50,900,500]);

line_styles = {'-o','-s','-^','-d'};
snr_x = flip(snr_levels);

for m = 1:4
    plot(snr_x, flip(robustness(m,:)), line_styles{m}, ...
        'Color', bar_colours(m,:), ...
        'LineWidth', 2.2, ...
        'MarkerSize', 9, ...
        'MarkerFaceColor', bar_colours(m,:));
    hold on;
end

xlabel('SNR (dB)  — Higher = Cleaner Signal', 'FontSize',12);
ylabel('Classification Accuracy (%)', 'FontSize',12);
title('Noise Robustness — Accuracy vs SNR Level', ...
      'FontSize',14,'FontWeight','bold');
legend(model_names,'Location','southeast','FontSize',11);
xticks(flip(snr_levels));
grid on; grid minor;
xlim([18, 52]);
ylim([70, 102]);
text(21, 72, '<-- More Noise', 'FontSize',10,'Color',[0.5 0 0]);
text(43, 72, 'Less Noise -->', 'FontSize',10,'Color',[0 0.4 0]);
yline(95,'k--','95%','LineWidth',1,'FontSize',10);

saveas(gcf,'../Figures/Stage5_NoiseRobustness.png');
fprintf('\n  Saved: Stage5_NoiseRobustness.png\n');

% =========================================================
% SECTION 5: Feature Importance (Random Forest)
% =========================================================
fprintf('\n--- Computing feature importance ---\n');

% FIX: Retrain a dedicated RF with OOBPredictorImportance ON
% The original mdl_rf was saved without this flag enabled.
% This retraining takes ~2 seconds and is only used for
% the importance plot — it does NOT affect any accuracy results.
fprintf('  Retraining RF with importance tracking (~2 sec)...\n');

mdl_rf_imp = TreeBagger(100, X_train, Y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on', ...   % <-- key flag
    'MinLeafSize', 1, ...
    'NumPredictorsToSample', round(sqrt(size(X_train,2))), ...
    'Reproducible', true);

importance = mdl_rf_imp.OOBPermutedPredictorDeltaError;
n_feats    = length(importance);

% Build feature names to match extraction order
feat_names    = {};
subband_names = {'D1','D2','D3','D4','D5','A5'};
stat_names    = {'Energy','Mean','Std','Skew','Kurt','Entropy'};
for sb = 1:6
    for st = 1:6
        feat_names{end+1} = sprintf('%s_%s', ...
            subband_names{sb}, stat_names{st});
    end
end
td_names = {'RMS','Peak','CrestF','THD','ZCR','FormF'};
for td = 1:length(td_names)
    feat_names{end+1} = td_names{td};
end
feat_names = feat_names(1:n_feats);

[imp_sorted, imp_idx] = sort(importance, 'descend');
top_n = min(20, n_feats);

figure('Name','Feature Importance','NumberTitle','off', ...
       'Position',[50,50,1100,520]);

barh(flip(imp_sorted(1:top_n)), ...
    'FaceColor',[0.11 0.45 0.18],'EdgeColor','none');
yticks(1:top_n);
yticklabels(flip(feat_names(imp_idx(1:top_n))));
xlabel('Importance Score (OOB Permutation)','FontSize',12);
title(sprintf('Top %d Most Important Features — Random Forest', top_n), ...
      'FontSize',14,'FontWeight','bold');
grid on; grid minor;

saveas(gcf,'../Figures/Stage5_FeatureImportance.png');
fprintf('  Saved: Stage5_FeatureImportance.png\n');

% =========================================================
% SECTION 6: Final 4-Panel Summary Dashboard
% =========================================================
figure('Name','Final Results Summary','NumberTitle','off', ...
       'Position',[50,50,1200,700]);

mean_f1  = mean(f1_scores, 2) * 100;
acc_drop = robustness(:,1) - robustness(:,4);

% Panel 1: Accuracy
subplot(2,2,1);
b2 = bar(accuracies,'FaceColor','flat');
for k=1:4, b2.CData(k,:)=bar_colours(k,:); end
for k=1:4
    text(k, accuracies(k)+0.2, sprintf('%.2f%%',accuracies(k)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
xticks(1:4); xticklabels(model_names); xtickangle(20);
ylabel('Accuracy (%)');
title('Classification Accuracy (SNR=40dB)');
ylim([min(accuracies)-5, 103]); grid on;
yline(95,'r--','LineWidth',1.2);

% Panel 2: Training Time
subplot(2,2,2);
b3 = bar(train_times,'FaceColor','flat');
for k=1:4, b3.CData(k,:)=bar_colours(k,:); end
for k=1:4
    text(k, train_times(k)+max(train_times)*0.02, ...
        sprintf('%.1fs',train_times(k)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
xticks(1:4); xticklabels(model_names); xtickangle(20);
ylabel('Seconds'); title('Training Time'); grid on;

% Panel 3: Mean F1
subplot(2,2,3);
b4 = bar(mean_f1,'FaceColor','flat');
for k=1:4, b4.CData(k,:)=bar_colours(k,:); end
for k=1:4
    text(k, mean_f1(k)+0.2, sprintf('%.2f%%',mean_f1(k)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
xticks(1:4); xticklabels(model_names); xtickangle(20);
ylabel('Mean F1 (%)'); title('Mean F1-Score (SNR=40dB)');
ylim([min(mean_f1)-5, 103]); grid on;

% Panel 4: Robustness Drop
subplot(2,2,4);
b5 = bar(acc_drop,'FaceColor','flat');
for k=1:4, b5.CData(k,:)=bar_colours(k,:); end
for k=1:4
    text(k, acc_drop(k)+0.1, sprintf('%.2f%%',acc_drop(k)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
xticks(1:4); xticklabels(model_names); xtickangle(20);
ylabel('Accuracy Drop (%)');
title('Robustness: Drop from 50dB to 20dB'); grid on;

sgtitle('FINAL RESULTS — Power Quality Classification Project', ...
        'FontSize',14,'FontWeight','bold');

saveas(gcf,'../Figures/Stage5_FinalSummary.png');

% =========================================================
% SECTION 7: Print Complete Final Report
% =========================================================
fprintf('\n');
fprintf('##############################################\n');
fprintf('##   FINAL PROJECT RESULTS — COMPLETE      ##\n');
fprintf('##############################################\n\n');

fprintf('DATASET\n');
fprintf('  Signals  : 4,000 (500 per class x 8 classes)\n');
fprintf('  Features : %d per signal\n', n_feats);
fprintf('  Split    : 70%% train / 30%% test\n');
fprintf('  Wavelet  : db4, 5-level DWT\n\n');

fprintf('CLASSIFIER RESULTS (SNR = 40 dB)\n');
fprintf('%-20s %10s %10s %10s %10s\n', ...
    'Model','Accuracy','Mean F1','Train(s)','Drop');
fprintf('%s\n', repmat('-',1,62));
for m = 1:4
    fprintf('%-20s %9.2f%% %9.2f%% %9.2fs %9.2f%%\n', ...
        model_names{m}, accuracies(m), mean_f1(m), ...
        train_times(m), acc_drop(m));
end
fprintf('%s\n', repmat('-',1,62));

[~,bi] = max(accuracies);
[~,fi] = max(mean_f1);
[~,ri] = min(acc_drop);
[~,ti] = min(train_times);
fprintf('\nBest Accuracy  : %s (%.2f%%)\n', model_names{bi}, accuracies(bi));
fprintf('Best F1        : %s (%.2f%%)\n',  model_names{fi}, mean_f1(fi));
fprintf('Most Robust    : %s (drop=%.2f%%)\n', model_names{ri}, acc_drop(ri));
fprintf('Fastest Train  : %s (%.2fs)\n',   model_names{ti}, train_times(ti));

fprintf('\nNOISE ROBUSTNESS\n');
fprintf('%-20s %8s %8s %8s %8s\n','Model','50dB','40dB','30dB','20dB');
fprintf('%s\n', repmat('-',1,52));
for m = 1:4
    fprintf('%-20s %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n', ...
        model_names{m}, robustness(m,1), robustness(m,2), ...
        robustness(m,3), robustness(m,4));
end

fprintf('\n##############################################\n');
fprintf('## ALL 5 STAGES COMPLETE — PROJECT DONE!  ##\n');
fprintf('##############################################\n');
fprintf('\nFigures saved to: ../Figures/\n');
fprintf('  Stage5_ConfusionMatrices_All.png\n');
fprintf('  Stage5_F1Scores.png\n');
fprintf('  Stage5_NoiseRobustness.png\n');
fprintf('  Stage5_FeatureImportance.png\n');
fprintf('  Stage5_FinalSummary.png\n');