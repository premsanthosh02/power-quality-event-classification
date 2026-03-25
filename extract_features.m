% =========================================================
% extract_features.m
% Stage 3: Feature Extraction using DWT + Statistics
% Project: Power Quality Classification in Industrial Plants
% =========================================================
% This script converts each preprocessed signal (2560 points)
% into a compact 36-dimensional feature vector.
%
% Feature breakdown:
%   DWT sub-bands (db4, 5 levels) × 6 stats = 30 features
%   Time-domain statistics                  =  6 features
%   TOTAL                                   = 36 features
%
% We extract features for ALL 4 SNR levels so we can test
% how well classifiers work under different noise conditions.
% =========================================================

clc;
clear;
close all;

fprintf('==============================================\n');
fprintf('  PQ Feature Extraction - Stage 3\n');
fprintf('==============================================\n\n');

% ---------------------------------------------------------
% SECTION 1: Parameters
% ---------------------------------------------------------
snr_levels   = [50, 40, 30, 20];  % Process all 4 noise levels
wavelet_name = 'db4';              % Daubechies-4 wavelet
                                   % Why db4? It closely resembles
                                   % power signal waveforms, making
                                   % it ideal for PQ analysis
decomp_level = 5;                  % Decompose into 5 levels
                                   % giving us D1,D2,D3,D4,D5 and A5

n_subbands   = decomp_level + 1;   % 5 detail + 1 approx = 6
n_stats      = 6;                  % Stats per sub-band
n_dwt_feats  = n_subbands * n_stats;  % 6×6 = 36... wait:
% Actually: 5 levels × 6 stats = 30, plus A5 × 6 = 6 → total 36
n_td_feats   = 6;                  % Time-domain features
n_features   = n_dwt_feats + n_td_feats;  % 30 + 6 = 36 total

fprintf('Feature extraction settings:\n');
fprintf('  Wavelet      : %s\n', wavelet_name);
fprintf('  Decomp levels: %d\n', decomp_level);
fprintf('  Sub-bands    : %d (D1-D5 + A5)\n', n_subbands);
fprintf('  Stats/band   : %d\n', n_stats);
fprintf('  DWT features : %d\n', n_dwt_feats);
fprintf('  Time-domain  : %d\n', n_td_feats);
fprintf('  TOTAL        : %d features per signal\n\n', n_features);

% ---------------------------------------------------------
% SECTION 2: Process Each SNR Level
% ---------------------------------------------------------

for snr_idx = 1 : length(snr_levels)

    snr_db = snr_levels(snr_idx);
    fprintf('--- Extracting features at SNR = %d dB ---\n', snr_db);

    % Load the preprocessed signals for this SNR level
    filename_in = sprintf('../Data/pq_preprocessed_SNR%d.mat', snr_db);
    load(filename_in);
    % Loads: signals_processed, labels, all_labels, fs, f0, etc.

    [n_signals, n_samples] = size(signals_processed);
    fprintf('  Signals loaded: %d × %d samples\n', n_signals, n_samples);

    % Pre-allocate the feature matrix
    % Each ROW = one signal's feature vector
    % Each COLUMN = one feature
    feature_matrix = zeros(n_signals, n_features);

    % ----- PROCESS EACH SIGNAL -------------------------
    for i = 1 : n_signals

        x = signals_processed(i, :);   % One signal: 1×2560

        % ================================================
        % PART A: DWT Feature Extraction
        % ================================================
        % wavedec() decomposes the signal into wavelet
        % coefficients at multiple frequency levels.
        %
        % Returns:
        %   C = all coefficients packed into one long vector
        %   L = length of each sub-band (to unpack C)

        [C, L] = wavedec(x, decomp_level, wavelet_name);

        % Extract each sub-band and compute 6 statistics
        feat_dwt = zeros(1, n_dwt_feats);
        feat_idx = 1;

        % --- Detail coefficients D5, D4, D3, D2, D1 ---
        % detcoef() extracts the detail coefficients at
        % a specific decomposition level
        for level = 1 : decomp_level
            d = detcoef(C, L, level);   % Detail at this level

            % Compute 6 statistics on this sub-band
            stats = compute_subband_stats(d);

            % Store in feature vector
            feat_dwt(feat_idx : feat_idx + n_stats - 1) = stats;
            feat_idx = feat_idx + n_stats;
        end

        % --- Approximation coefficients A5 -------------
        % appcoef() extracts the lowest-frequency band
        % A5 contains the fundamental frequency region
        a5 = appcoef(C, L, wavelet_name, decomp_level);
        stats_a5 = compute_subband_stats(a5);
        feat_dwt(feat_idx : feat_idx + n_stats - 1) = stats_a5;

        % ================================================
        % PART B: Time-Domain Feature Extraction
        % ================================================
        feat_td = compute_time_domain_features(x, fs, f0);

        % ================================================
        % PART C: Combine into Final Feature Vector
        % ================================================
        feature_matrix(i, :) = [feat_dwt, feat_td];

    end

    fprintf('  Feature matrix: %d × %d\n', ...
        size(feature_matrix,1), size(feature_matrix,2));

    % Save the feature matrix for this SNR level
    filename_out = sprintf('../Data/pq_features_SNR%d.mat', snr_db);
    save(filename_out, 'feature_matrix', 'labels', 'all_labels', ...
         'fs', 'f0', 'class_names', 'num_per_class', 'snr_db', ...
         'n_features', 'wavelet_name', 'decomp_level');

    fprintf('  Saved: pq_features_SNR%d.mat\n\n', snr_db);

end

% ---------------------------------------------------------
% SECTION 3: Feature Analysis & Visualisation
% ---------------------------------------------------------
% Load SNR=40 features for visualisation (good balance)
load('../Data/pq_features_SNR40.mat');

n_signals = size(feature_matrix, 1);
fprintf('Generating feature analysis plots...\n');

% ---- Plot 1: Feature vector heatmap -------------------
% Shows all 36 features for the first 80 signals (10 per class)
% Rows = signals, Columns = features
% Colour shows feature value (dark=low, bright=high)

figure('Name','Feature Matrix Heatmap','NumberTitle','off',...
       'Position',[50,50,1200,500]);

% Take first 10 signals per class = 80 signals total
n_show = 10;
idx_show = [];
for c = 0:7
    rows = find(all_labels == c, n_show);
    idx_show = [idx_show; rows(1:min(n_show,length(rows)))];
end

% Normalise each feature column to [0,1] for display
F_show = feature_matrix(idx_show, :);
F_norm = (F_show - min(F_show)) ./ (max(F_show) - min(F_show) + 1e-10);

imagesc(F_norm);
colormap(jet);
colorbar;
xlabel('Feature Index (1–36)', 'FontSize', 12);
ylabel('Signal Index', 'FontSize', 12);
title('Feature Matrix Heatmap — 10 Samples per Class (SNR=40dB)', ...
      'FontSize', 13, 'FontWeight', 'bold');

% Add class boundary lines
hold on;
for c = 1:7
    yline(c*n_show + 0.5, 'w-', 'LineWidth', 1.5);
end
% Add vertical line separating DWT and time-domain features
xline(30.5, 'w--', 'LineWidth', 2);
text(15, 82, 'DWT Features (1-30)', 'Color','white', ...
    'FontSize', 10, 'HorizontalAlignment','center');
text(33, 82, 'TD (31-36)', 'Color','white', ...
    'FontSize', 9, 'HorizontalAlignment','center');

% Class labels on Y axis
ytick_pos = (1:8) * n_show - n_show/2;
yticks(ytick_pos);
yticklabels(class_names);

saveas(gcf, '../Figures/Stage3_FeatureHeatmap.png');

% ---- Plot 2: Feature values per class (boxplot) -------
% Shows distribution of selected key features per class
% Helps verify features are discriminating between classes

figure('Name','Feature Distribution per Class','NumberTitle','off',...
       'Position',[50,50,1400,600]);

% Pick 6 representative features to show
feat_labels_all = {};
for i = 1:5
    feat_labels_all{end+1} = sprintf('D%d Energy',i);
    feat_labels_all{end+1} = sprintf('D%d Mean',i);
    feat_labels_all{end+1} = sprintf('D%d Std',i);
    feat_labels_all{end+1} = sprintf('D%d Skew',i);
    feat_labels_all{end+1} = sprintf('D%d Kurt',i);
    feat_labels_all{end+1} = sprintf('D%d Entr',i);
end
feat_labels_all{end+1} = 'A5 Energy';
feat_labels_all{end+1} = 'A5 Mean';
feat_labels_all{end+1} = 'A5 Std';
feat_labels_all{end+1} = 'A5 Skew';
feat_labels_all{end+1} = 'A5 Kurt';
feat_labels_all{end+1} = 'A5 Entr';
feat_labels_all{end+1} = 'RMS';
feat_labels_all{end+1} = 'Peak';
feat_labels_all{end+1} = 'Crest F';
feat_labels_all{end+1} = 'THD';
feat_labels_all{end+1} = 'ZCR';
feat_labels_all{end+1} = 'Form F';

% Show 6 key features: D1 Energy, D3 Energy, A5 Energy,
% RMS, THD, Crest Factor
key_feats   = [1, 13, 31, 33, 34, 36];
key_names   = {'D1 Energy','D3 Energy','A5 Energy', ...
               'RMS','Peak','Form Factor'};
colours_cls = lines(8);

for p = 1:6
    subplot(2, 3, p);
    feat_col = key_feats(p);

    data_per_class = zeros(num_per_class, 8);
    for c = 0:7
        idx_c = find(all_labels == c);
        vals  = feature_matrix(idx_c, feat_col);
        n_use = min(length(vals), num_per_class);
        data_per_class(1:n_use, c+1) = vals(1:n_use);
    end

    boxplot(data_per_class, 'Labels', class_names, ...
            'Colors', 'k', 'Symbol', '');
    title(key_names{p}, 'FontWeight','bold', 'FontSize',11);
    xlabel('PQ Class'); ylabel('Feature Value');
    xtickangle(30);
    grid on;

    % Colour the boxes
    h = findobj(gca,'Tag','Box');
    for j = 1:length(h)
        patch(get(h(j),'XData'), get(h(j),'YData'), ...
              colours_cls(length(h)+1-j,:), 'FaceAlpha',0.5);
    end
end

sgtitle('Key Feature Distributions per PQ Class (SNR=40dB)', ...
        'FontSize',13,'FontWeight','bold');

saveas(gcf,'../Figures/Stage3_FeatureDistributions.png');

% ---- Plot 3: DWT Decomposition for one signal ---------
% Shows what the wavelet decomposition looks like visually
% for one example signal (Harmonics class)

load('../Data/pq_preprocessed_SNR40.mat');
harm_idx = find(all_labels == 4, 1);   % First Harmonics signal
x_harm   = signals_processed(harm_idx, :);
t_ms     = (0 : length(x_harm)-1) / fs * 1000;

[C_h, L_h] = wavedec(x_harm, decomp_level, wavelet_name);

figure('Name','DWT Decomposition — Harmonics Signal', ...
       'NumberTitle','off','Position',[50,50,1200,800]);

% Plot original signal at top
subplot(7, 1, 1);
plot(t_ms, x_harm, 'Color','#1A4FA0','LineWidth',1);
title('Original Signal — Harmonics (SNR=40dB)', ...
      'FontWeight','bold','FontSize',11);
ylabel('Amp'); grid on;

% Plot each detail sub-band
sub_colours = {'#AD1457','#E65100','#F9A825', ...
               '#2E7D32','#1565C0','#4A148C'};
sub_names   = {'D1 (3.2–6.4 kHz)','D2 (1.6–3.2 kHz)', ...
               'D3 (800–1600 Hz)','D4 (400–800 Hz)', ...
               'D5 (200–400 Hz)','A5 (0–200 Hz)'};

for lv = 1:5
    subplot(7, 1, lv+1);
    d    = detcoef(C_h, L_h, lv);
    t_d  = linspace(0, max(t_ms), length(d));
    plot(t_d, d, 'Color', sub_colours{lv}, 'LineWidth', 0.8);
    ylabel(sprintf('D%d',lv),'FontSize',9);
    title(sub_names{lv},'FontSize',9);
    grid on;
end

% Plot approximation A5
subplot(7, 1, 7);
a5_h = appcoef(C_h, L_h, wavelet_name, decomp_level);
t_a5 = linspace(0, max(t_ms), length(a5_h));
plot(t_a5, a5_h, 'Color', sub_colours{6}, 'LineWidth', 1);
ylabel('A5','FontSize',9);
title(sub_names{6},'FontSize',9);
xlabel('Time (ms)'); grid on;

sgtitle('DWT Decomposition — db4, 5 Levels (Harmonics Signal)', ...
        'FontSize',13,'FontWeight','bold');

saveas(gcf,'../Figures/Stage3_DWT_Decomposition.png');

% ---------------------------------------------------------
% SECTION 4: Print Feature Summary
% ---------------------------------------------------------
load('../Data/pq_features_SNR40.mat');

fprintf('\n==============================================\n');
fprintf('  Feature Summary (SNR = 40 dB)\n');
fprintf('==============================================\n');
fprintf('Feature matrix shape: %d × %d\n', ...
    size(feature_matrix,1), size(feature_matrix,2));
fprintf('\nFeature breakdown:\n');
fprintf('  Features  1–30 : DWT sub-band statistics\n');
fprintf('    D1 (3.2–6.4 kHz) : Features  1– 6\n');
fprintf('    D2 (1.6–3.2 kHz) : Features  7–12\n');
fprintf('    D3 (0.8–1.6 kHz) : Features 13–18\n');
fprintf('    D4 (0.4–0.8 kHz) : Features 19–24\n');
fprintf('    D5 (0.2–0.4 kHz) : Features 25–30\n');
fprintf('    A5 (0 –0.2 kHz)  : Features 25–30\n');
fprintf('  Features 31–36 : Time-domain statistics\n');
fprintf('    31: RMS  32: Peak  33: Crest Factor\n');
fprintf('    34: THD  35: ZCR   36: Form Factor\n');
fprintf('\nSample feature vector (Signal 1):\n');
disp(feature_matrix(1,:));
fprintf('\nReady for Stage 4: Model Training\n');
fprintf('==============================================\n');

% =========================================================
% LOCAL HELPER FUNCTIONS
% (defined at the bottom of the script — MATLAB requires
%  all functions to be at the END of a script file)
% =========================================================

% ---------------------------------------------------------
% Function: compute_subband_stats
% Computes 6 statistical descriptors from a set of
% wavelet coefficients (one sub-band)
% Input : coeffs — vector of wavelet coefficients
% Output: stats  — 1×6 vector [energy, mean, std,
%                               skewness, kurtosis, entropy]
% ---------------------------------------------------------
function stats = compute_subband_stats(coeffs)

    coeffs = double(coeffs(:)');    % Ensure row vector, double

    % 1. Energy — total signal power in this frequency band
    %    High energy in D1 → noise/transient
    %    High energy in A5 → fundamental present
    energy = sum(coeffs .^ 2);

    % 2. Mean — average value of coefficients
    %    Non-zero mean can indicate waveform asymmetry
    mu = mean(coeffs);

    % 3. Standard Deviation — spread of coefficients
    %    High std in D3 → harmonic content present
    sigma = std(coeffs);

    % 4. Skewness — asymmetry of coefficient distribution
    %    Zero for symmetric (Normal), nonzero for sag/swell
    sk = skewness(coeffs);

    % 5. Kurtosis — peakedness of distribution
    %    Very high kurtosis → impulsive event (transient)
    ku = kurtosis(coeffs);

    % 6. Shannon Entropy — measure of information/disorder
    %    Low entropy → organised (Normal)
    %    High entropy → disordered (noisy, complex events)
    %    We use log-energy entropy formula:
    %    H = -sum(c² × log(c²))  for non-zero coefficients
    c2 = coeffs .^ 2;
    c2_nz = c2(c2 > 0);          % Ignore zeros (log(0) = -inf)
    if isempty(c2_nz)
        ent = 0;
    else
        ent = -sum(c2_nz .* log(c2_nz));
    end

    stats = [energy, mu, sigma, sk, ku, ent];

end

% ---------------------------------------------------------
% Function: compute_time_domain_features
% Computes 6 time-domain features directly from raw signal
% Input : x  — signal vector
%         fs — sampling frequency (Hz)
%         f0 — fundamental frequency (Hz)
% Output: td — 1×6 vector
% ---------------------------------------------------------
function td = compute_time_domain_features(x, fs, f0)

    x = double(x(:)');    % Ensure row vector

    % 1. RMS (Root Mean Square)
    %    The effective voltage — standard power measurement
    %    RMS drops during sag, rises during swell
    rms_val = sqrt(mean(x .^ 2));

    % 2. Peak Value — maximum absolute amplitude
    %    Very high peak → transient or swell
    peak_val = max(abs(x));

    % 3. Crest Factor = Peak / RMS
    %    Pure sine: crest factor = sqrt(2) ≈ 1.414
    %    Impulsive transient: crest factor >> 1.414
    %    Clipped/interrupted: crest factor < 1.414
    if rms_val > 0
        crest = peak_val / rms_val;
    else
        crest = 0;
    end

    % 4. Total Harmonic Distortion (THD) in percent
    %    Measures how much harmonic content exists
    %    Normal: THD ≈ 0%
    %    Harmonics class: THD >> 0%
    thd_val = compute_thd(x, f0, fs);

    % 5. Zero Crossing Rate (ZCR)
    %    How many times the signal crosses zero per second
    %    Normal sine at 50 Hz: ZCR ≈ 100 crossings/sec (2/cycle)
    %    Harmonics distort the waveform → more zero crossings
    n = length(x);
    zero_crossings = sum(abs(diff(sign(x)))) / 2;
    zcr = zero_crossings / (n / fs);  % Crossings per second

    % 6. Form Factor = RMS / Mean of absolute value
    %    Pure sine: form factor = pi/(2*sqrt(2)) ≈ 1.1107
    %    Changes with distortion type
    mean_abs = mean(abs(x));
    if mean_abs > 0
        form_f = rms_val / mean_abs;
    else
        form_f = 0;
    end

    td = [rms_val, peak_val, crest, thd_val, zcr, form_f];

end

% ---------------------------------------------------------
% Function: compute_thd
% Computes Total Harmonic Distortion as a percentage
% THD = sqrt(sum of harmonic powers) / fundamental power × 100
% Input : x  — signal
%         f0 — fundamental frequency
%         fs — sampling rate
% Output: thd — THD percentage
% ---------------------------------------------------------
function thd = compute_thd(x, f0, fs)

    N     = length(x);
    X_fft = abs(fft(x) / N);    % FFT magnitude, normalised
    freqs = (0:N-1) * fs / N;   % Frequency axis

    % Find the index of the fundamental frequency
    [~, fund_idx] = min(abs(freqs - f0));
    V1 = X_fft(fund_idx);        % Fundamental voltage magnitude

    if V1 < 1e-10
        thd = 0;
        return;
    end

    % Sum power of harmonics 2nd through 7th
    harm_power = 0;
    for h = 2:7
        harm_freq = h * f0;
        [~, h_idx] = min(abs(freqs - harm_freq));
        if h_idx <= N/2
            harm_power = harm_power + X_fft(h_idx)^2;
        end
    end

    thd = (sqrt(harm_power) / V1) * 100;  % As percentage

end
