% =========================================================
% preprocess_signals.m
% Stage 2: Signal Preprocessing
% Project: Power Quality Classification in Industrial Plants
% =========================================================
% This script takes the raw generated signals and prepares
% them for feature extraction by:
%   1. Normalising amplitude to [-1, +1]
%   2. Adding realistic Gaussian noise (4 SNR levels)
%   3. Removing DC offset via high-pass filter
%   4. Plotting before vs after comparison
% =========================================================

clc;
clear;
close all;

fprintf('==============================================\n');
fprintf('  PQ Signal Preprocessing - Stage 2\n');
fprintf('==============================================\n\n');

% ---------------------------------------------------------
% SECTION 1: Load the Dataset from Stage 1
% ---------------------------------------------------------
fprintf('Loading dataset from Stage 1...\n');

load('../Data/pq_dataset.mat');
% This loads: all_signals, labels, all_labels, fs, f0,
%             N, cycles, n_pts, class_names, num_per_class

fprintf('Loaded: %d signals × %d samples\n\n', ...
    size(all_signals,1), size(all_signals,2));

% ---------------------------------------------------------
% SECTION 2: Define the Preprocessing Function
% ---------------------------------------------------------
% We will preprocess at 4 different noise levels
% SNR = Signal-to-Noise Ratio in decibels (dB)
%
% What is SNR?
%   SNR 50 dB = very clean signal, almost no noise
%   SNR 40 dB = lightly noisy (good lab conditions)
%   SNR 30 dB = moderately noisy (typical factory)
%   SNR 20 dB = heavily noisy (harsh industrial environment)
%
% Higher dB = cleaner signal
% Lower dB  = noisier signal

snr_levels = [50, 40, 30, 20];   % dB values to test

% We will save one preprocessed dataset per SNR level
% These will be used in Stage 4 to test noise robustness

% ---------------------------------------------------------
% SECTION 3: Show "Before" — Raw Signal Properties
% ---------------------------------------------------------
fprintf('--- Raw Signal Statistics (Before Preprocessing) ---\n');

% Check amplitude range of first few signals
amp_max = max(max(abs(all_signals(1:100, :))));
amp_min = min(min(abs(all_signals(1:100, :))));
fprintf('Raw amplitude range: [%.4f, %.4f]\n', amp_min, amp_max);

% Check if any DC offset exists
dc_sample = mean(all_signals(1, :));
fprintf('DC offset example (signal 1): %.6f\n\n', dc_sample);

% ---------------------------------------------------------
% SECTION 4: The Preprocessing Pipeline
% ---------------------------------------------------------
% This function processes ALL signals at a chosen SNR level

% Design the high-pass filter ONCE (reuse for all signals)
% Butterworth filter, 2nd order, cutoff at 5 Hz
% Purpose: remove DC offset and very slow drift
% Cutoff 5 Hz means: frequencies below 5 Hz are removed
%                    frequencies above 5 Hz pass through
% This keeps our 50 Hz fundamental and all harmonics intact

filter_cutoff = 5;       % Hz — remove anything below this
filter_order  = 2;       % 2nd order Butterworth (gentle roll-off)

% Normalise cutoff to [0,1] where 1 = Nyquist frequency (fs/2)
Wn = filter_cutoff / (fs / 2);

% Design the filter (returns filter coefficients b, a)
[b_hpf, a_hpf] = butter(filter_order, Wn, 'high');

fprintf('High-pass filter designed:\n');
fprintf('  Type   : Butterworth (2nd order)\n');
fprintf('  Cutoff : %d Hz\n\n', filter_cutoff);

% ---------------------------------------------------------
% MAIN PREPROCESSING LOOP
% Processes signals at each SNR level and saves results
% ---------------------------------------------------------

for snr_idx = 1 : length(snr_levels)

    snr_db = snr_levels(snr_idx);
    fprintf('Processing at SNR = %d dB...\n', snr_db);

    [n_signals, n_samples] = size(all_signals);
    signals_processed = zeros(n_signals, n_samples);  % Pre-allocate

    for i = 1 : n_signals

        x = all_signals(i, :);   % Get one raw signal (1×2560)

        % ---- STEP 1: Normalise -------------------------
        % Scale the signal so its maximum absolute value = 1.0
        % This ensures no single class has artificially larger
        % amplitude that could bias the classifier
        peak = max(abs(x));
        if peak > 0
            x = x / peak;        % Now x is in range [-1, +1]
        end

        % ---- STEP 2: Add Gaussian White Noise ----------
        % awgn() = Additive White Gaussian Noise
        % MATLAB built-in from Signal Processing Toolbox
        % 'measured' means SNR is relative to signal power
        x = awgn(x, snr_db, 'measured');

        % ---- STEP 3: High-Pass Filter ------------------
        % filtfilt() applies the filter FORWARD and BACKWARD
        % This gives zero phase distortion (no time shift)
        % Important: we don't want to shift the signal in time!
        x = filtfilt(b_hpf, a_hpf, x);

        % Store the processed signal
        signals_processed(i, :) = x;

    end

    % Save this SNR level's processed dataset
    filename = sprintf('../Data/pq_preprocessed_SNR%d.mat', snr_db);
    save(filename, 'signals_processed', 'labels', 'all_labels', ...
         'fs', 'f0', 'N', 'cycles', 'n_pts', 'class_names', ...
         'num_per_class', 'snr_db');

    fprintf('  Saved: %s\n', filename);

end

fprintf('\nAll SNR levels processed and saved.\n\n');

% ---------------------------------------------------------
% SECTION 5: Visual Comparison — Before vs After
% ---------------------------------------------------------
% Show one signal (Normal class) at all preprocessing stages
% so you can SEE what each step does to the waveform

fprintf('Generating comparison plots...\n');

% Use the first Normal signal for demonstration
demo_raw = all_signals(1, :);     % Raw signal (Class 0: Normal)
t_ms     = (0 : n_pts-1) / fs * 1000;   % Time in milliseconds

% Recreate preprocessing steps individually for plotting
demo_norm   = demo_raw / max(abs(demo_raw));             % After normalise
demo_noisy  = awgn(demo_norm, 30, 'measured');           % After noise (30dB)
demo_filter = filtfilt(b_hpf, a_hpf, demo_noisy);       % After filter

% ---- Plot 1: 4-step comparison for NORMAL signal --------
figure('Name', 'Preprocessing Steps — Normal Signal', ...
       'NumberTitle', 'off', ...
       'Position', [50, 50, 1300, 700]);

subplot(2, 2, 1);
plot(t_ms, demo_raw, 'Color', '#1565C0', 'LineWidth', 1.2);
title('Step 0: Raw Signal (Original)', 'FontWeight', 'bold');
xlabel('Time (ms)'); ylabel('Amplitude');
ylim([-1.5 1.5]); grid on; grid minor;

subplot(2, 2, 2);
plot(t_ms, demo_norm, 'Color', '#2E7D32', 'LineWidth', 1.2);
title('Step 1: After Normalisation (÷ peak)', 'FontWeight', 'bold');
xlabel('Time (ms)'); ylabel('Amplitude (pu)');
ylim([-1.5 1.5]); grid on; grid minor;

subplot(2, 2, 3);
plot(t_ms, demo_noisy, 'Color', '#E65100', 'LineWidth', 0.8);
title('Step 2: After Adding Noise (SNR = 30 dB)', 'FontWeight', 'bold');
xlabel('Time (ms)'); ylabel('Amplitude (pu)');
ylim([-1.5 1.5]); grid on; grid minor;

subplot(2, 2, 4);
plot(t_ms, demo_filter, 'Color', '#6A1B9A', 'LineWidth', 1.2);
title('Step 3: After High-Pass Filter (DC removed)', 'FontWeight', 'bold');
xlabel('Time (ms)'); ylabel('Amplitude (pu)');
ylim([-1.5 1.5]); grid on; grid minor;

sgtitle('Preprocessing Pipeline — Step by Step (Normal Signal, SNR=30dB)', ...
        'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, '../Figures/Stage2_PreprocessingSteps.png');

% ---- Plot 2: Same SAG signal at all 4 SNR levels --------
figure('Name', 'SNR Comparison — Sag Signal', ...
       'NumberTitle', 'off', ...
       'Position', [50, 50, 1300, 700]);

% Get a sag signal (first sag = row 2 in our dataset)
sag_raw  = all_signals(2, :);
sag_norm = sag_raw / max(abs(sag_raw));

snr_colours = {'#1565C0', '#2E7D32', '#E65100', '#C62828'};
snr_titles  = {'SNR = 50 dB (Near Clean)', ...
                'SNR = 40 dB (Light Noise)', ...
                'SNR = 30 dB (Moderate Noise)', ...
                'SNR = 20 dB (Heavy Noise)'};

for s = 1:4
    subplot(2, 2, s);
    sag_noisy    = awgn(sag_norm, snr_levels(s), 'measured');
    sag_filtered = filtfilt(b_hpf, a_hpf, sag_noisy);
    plot(t_ms, sag_filtered, 'Color', snr_colours{s}, 'LineWidth', 1.0);
    title(snr_titles{s}, 'FontWeight', 'bold', 'FontSize', 11);
    xlabel('Time (ms)'); ylabel('Amplitude (pu)');
    ylim([-1.5 1.5]); grid on; grid minor;
    % Mark where the sag is visible
    xline(40, 'k--', 'Sag region', 'LabelVerticalAlignment', 'bottom');
end

sgtitle('Voltage Sag at 4 Noise Levels — After Preprocessing', ...
        'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, '../Figures/Stage2_SNR_Comparison.png');

% ---- Plot 3: All 8 classes after preprocessing (SNR=30) -
load('../Data/pq_preprocessed_SNR30.mat');

figure('Name', 'All 8 Classes After Preprocessing (SNR=30dB)', ...
       'NumberTitle', 'off', ...
       'Position', [100, 100, 1400, 700]);

colours = {'#2196F3','#FF5722','#4CAF50','#9C27B0', ...
           '#FF9800','#F44336','#009688','#3F51B5'};

first_idx = 1 : 8;
for c = 1 : 8
    subplot(2, 4, c);
    plot(t_ms, signals_processed(first_idx(c), :), ...
         'Color', colours{c}, 'LineWidth', 1.0);
    title(class_names{c}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time (ms)'); ylabel('Amplitude (pu)');
    ylim([-1.5 1.5]); grid on; grid minor;
end

sgtitle('All 8 PQ Classes — After Preprocessing (SNR = 30 dB)', ...
        'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, '../Figures/Stage2_AllClasses_Preprocessed.png');

% ---------------------------------------------------------
% SECTION 6: Print Summary
% ---------------------------------------------------------
fprintf('\n==============================================\n');
fprintf('  STAGE 2 COMPLETE — Summary\n');
fprintf('==============================================\n');
fprintf('Preprocessing steps applied:\n');
fprintf('  1. Normalisation    : amplitude → [-1, +1]\n');
fprintf('  2. Noise injection  : Gaussian white noise\n');
fprintf('  3. HP Filter        : Butterworth 2nd order, 5 Hz\n\n');
fprintf('Files saved:\n');
for s = 1:length(snr_levels)
    fprintf('  pq_preprocessed_SNR%d.mat\n', snr_levels(s));
end
fprintf('\nFigures saved:\n');
fprintf('  Stage2_PreprocessingSteps.png\n');
fprintf('  Stage2_SNR_Comparison.png\n');
fprintf('  Stage2_AllClasses_Preprocessed.png\n');
fprintf('\nReady for Stage 3: Feature Extraction\n');
fprintf('==============================================\n');