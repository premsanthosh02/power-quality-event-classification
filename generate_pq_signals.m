% =========================================================
% generate_pq_signals.m
% Stage 1: Generate all 8 PQ Event Classes
% Project: Power Quality Classification in Industrial Plants
% =========================================================
% This script creates a synthetic dataset of power quality
% signals using IEEE 1159 mathematical models.
% No hardware needed — everything is simulated.
% =========================================================

clc;          % Clear the command window
clear;        % Remove all variables from workspace
close all;    % Close any open figure windows

fprintf('==============================================\n');
fprintf('  PQ Signal Generation - Stage 1\n');
fprintf('==============================================\n\n');

% ---------------------------------------------------------
% SECTION 1: Define Signal Parameters
% ---------------------------------------------------------
% These values follow the IEEE 1159 standard for PQ monitoring

fs      = 12800;    % Sampling frequency in Hz
                    % (12800 samples per second)
                    % Why 12800? = 256 samples × 50 Hz
                    % This captures harmonics up to 6400 Hz

f0      = 50;       % Fundamental frequency = 50 Hz
                    % (standard in India, Europe, etc.)

N       = fs / f0;  % Samples per one cycle = 256
                    % One full sine wave = 256 data points

cycles  = 10;       % We record 10 complete cycles per signal
                    % Duration = 10/50 = 0.2 seconds

n_pts   = N * cycles;   % Total samples per signal = 2560

% Create the time vector (one row of time values)
% t goes from 0 to just before 0.2 seconds
% in steps of 1/12800 seconds
t = (0 : n_pts - 1) / fs;   % t is a 1×2560 vector

num_per_class = 500;  % How many signals to generate per class
                      % 500 × 8 classes = 4000 total signals

% Pre-allocate storage arrays for speed
% Each row = one signal (2560 values)
% Each column = one time sample
n_classes   = 8;
n_total     = num_per_class * n_classes;
all_signals = zeros(n_total, n_pts);  % 4000 × 2560 matrix
all_labels  = zeros(n_total, 1);      % 4000 × 1 label vector

fprintf('Signal parameters:\n');
fprintf('  Sampling rate : %d Hz\n', fs);
fprintf('  Fundamental   : %d Hz\n', f0);
fprintf('  Samples/cycle : %d\n', N);
fprintf('  Cycles/signal : %d\n', cycles);
fprintf('  Total samples per signal: %d\n', n_pts);
fprintf('  Signals per class: %d\n', num_per_class);
fprintf('  Total signals: %d\n\n', n_total);

% ---------------------------------------------------------
% SECTION 2: Generate Signals — One Class at a Time
% ---------------------------------------------------------

sig_idx = 1;  % Running index to fill all_signals row by row

for k = 1 : num_per_class

    % === CLASS 0: NORMAL (Pure Sine Wave) ================
    % The ideal voltage waveform — no disturbance at all.
    % This is the reference "healthy" signal.
    A = 1.0;
    x0 = A * sin(2 * pi * f0 * t);
    all_signals(sig_idx, :) = x0;
    all_labels(sig_idx) = 0;
    sig_idx = sig_idx + 1;

    % === CLASS 1: VOLTAGE SAG ============================
    % A temporary reduction in voltage.
    % Caused by: motor startups, faults on parallel feeders.
    % The voltage drops to (1-alpha) of normal for a short time.
    %
    % Formula: x(t) = [1 - alpha*(u(t-t1) - u(t-t2))] * sin(...)
    % where u() is the unit step function (heaviside)
    % alpha  = sag depth (0.1 to 0.9 per unit)
    % t1, t2 = start and end of the sag event

    alpha = 0.1 + rand * 0.8;          % Random depth each sample
    t1    = 0.02 + rand * 0.06;        % Random start time
    t2    = t1 + 0.02 + rand * 0.06;  % End time (always after t1)
    
    % heaviside(t - t1) = 0 before t1, 1 after t1
    % Difference = pulse that is 1 only between t1 and t2
    sag_envelope = 1 - alpha * (heaviside(t - t1) - heaviside(t - t2));
    x1 = A * sag_envelope .* sin(2 * pi * f0 * t);
    % Note: .* means element-wise multiplication
    
    all_signals(sig_idx, :) = x1;
    all_labels(sig_idx) = 1;
    sig_idx = sig_idx + 1;

    % === CLASS 2: VOLTAGE SWELL ==========================
    % A temporary INCREASE in voltage above normal.
    % Caused by: sudden load disconnection, capacitor switching.
    % Dangerous: can damage insulation and sensitive equipment.
    %
    % Same formula as sag but we ADD alpha instead of subtracting

    alpha = 0.1 + rand * 0.7;
    t1    = 0.02 + rand * 0.06;
    t2    = t1 + 0.02 + rand * 0.06;
    
    swell_envelope = 1 + alpha * (heaviside(t - t1) - heaviside(t - t2));
    x2 = A * swell_envelope .* sin(2 * pi * f0 * t);
    
    all_signals(sig_idx, :) = x2;
    all_labels(sig_idx) = 2;
    sig_idx = sig_idx + 1;

    % === CLASS 3: INTERRUPTION ===========================
    % Nearly complete loss of voltage (> 90% drop).
    % The most severe PQ event — stops all production.
    % Same as sag but alpha is close to 1.0

    alpha = 0.9 + rand * 0.1;          % 90–100% voltage loss
    t1    = 0.02 + rand * 0.06;
    t2    = t1 + 0.02 + rand * 0.04;
    
    intr_envelope = 1 - alpha * (heaviside(t - t1) - heaviside(t - t2));
    x3 = A * intr_envelope .* sin(2 * pi * f0 * t);
    
    all_signals(sig_idx, :) = x3;
    all_labels(sig_idx) = 3;
    sig_idx = sig_idx + 1;

    % === CLASS 4: HARMONICS ==============================
    % Waveform distortion caused by non-linear loads
    % (VFDs, UPS, rectifiers, LED drivers).
    % Extra sine waves at 3×, 5×, 7× the fundamental frequency
    % are added ON TOP of the fundamental.
    %
    % Formula: x(t) = a1*sin(2πf0t) + a3*sin(6πf0t) + a5*sin(10πf0t)

    a1 = 1.0;                      % Fundamental amplitude
    a3 = 0.15 + rand * 0.15;       % 3rd harmonic (150 Hz): 15–30%
    a5 = 0.05 + rand * 0.10;       % 5th harmonic (250 Hz): 5–15%
    a7 = 0.03 + rand * 0.05;       % 7th harmonic (350 Hz): 3–8%
    
    x4 = A * (a1 * sin(2*pi*f0*t) + ...
               a3 * sin(6*pi*f0*t) + ...     % 3rd harmonic = 3×50 = 150 Hz
               a5 * sin(10*pi*f0*t) + ...    % 5th harmonic = 5×50 = 250 Hz
               a7 * sin(14*pi*f0*t));         % 7th harmonic = 7×50 = 350 Hz
    
    all_signals(sig_idx, :) = x4;
    all_labels(sig_idx) = 4;
    sig_idx = sig_idx + 1;

    % === CLASS 5: TRANSIENT ==============================
    % A very short, high-energy voltage spike.
    % Caused by: lightning, switching operations, capacitor banks.
    % Duration < 0.5 ms. Can permanently destroy semiconductors.
    %
    % Formula: base sine + B*exp(-decay).*sin(high_freq) at t_occ

    t_occ = 0.02 + rand * 0.14;    % Random time of occurrence
    fn    = 300 + rand * 300;       % Transient frequency: 300–600 Hz
    tau   = 0.005 + rand * 0.005;   % Decay time constant
    B     = 0.3 + rand * 0.5;       % Transient amplitude
    
    % The transient only exists AFTER t_occ (use logical mask)
    transient = B * exp(-(t - t_occ)/tau) .* ...
                sin(2*pi*fn*(t - t_occ)) .* (t >= t_occ);
    
    x5 = A * sin(2*pi*f0*t) + transient;
    
    all_signals(sig_idx, :) = x5;
    all_labels(sig_idx) = 5;
    sig_idx = sig_idx + 1;

    % === CLASS 6: FLICKER ================================
    % Rapid, repetitive voltage fluctuations at low frequency.
    % Caused by: arc furnaces, welding machines, wind turbines.
    % Human eye detects flicker at 8–10 Hz as irritating lamp flicker.
    %
    % Formula: x(t) = [1 + af*sin(2π*ff*t)] * sin(2π*f0*t)
    % The outer sine (ff) slowly modulates the amplitude

    af = 0.05 + rand * 0.10;        % Flicker depth: 5–15%
    ff = 5 + rand * 25;             % Flicker frequency: 5–30 Hz
    
    flicker_envelope = 1 + af * sin(2 * pi * ff * t);
    x6 = A * flicker_envelope .* sin(2 * pi * f0 * t);
    
    all_signals(sig_idx, :) = x6;
    all_labels(sig_idx) = 6;
    sig_idx = sig_idx + 1;

    % === CLASS 7: NOTCHING ================================
    % Periodic short dips occurring once per cycle.
    % Caused by: power converters (when current commutates between phases).
    % Creates interference in control circuits and communication systems.

    x7 = A * sin(2 * pi * f0 * t);  % Start with pure sine
    
    notch_width  = 5;        % Notch width in number of samples
    notch_depth  = 0.8 + rand * 0.15;   % How deep the notch goes
    t1_notch     = 0.02 + rand * 0.04;  % Start of notching
    
    % Find the zero-crossings (positive-going) from t1 onwards
    % and place a notch at each one
    cycle_samples = N;   % 256 samples per cycle
    start_sample  = round(t1_notch * fs);
    
    idx = start_sample : cycle_samples : n_pts - notch_width;
    for m = idx
        if m > 0 && m + notch_width - 1 <= n_pts
            x7(m : m + notch_width - 1) = ...
                x7(m : m + notch_width - 1) * (1 - notch_depth);
        end
    end
    
    all_signals(sig_idx, :) = x7;
    all_labels(sig_idx) = 7;
    sig_idx = sig_idx + 1;

end

fprintf('Signal generation complete!\n');
fprintf('Dataset size: %d signals × %d samples\n\n', ...
    size(all_signals,1), size(all_signals,2));

% ---------------------------------------------------------
% SECTION 3: Convert Labels to Categorical
% ---------------------------------------------------------
% Categorical is MATLAB's way of storing class labels
% like 'Sag', 'Swell', etc. instead of just numbers 0-7

class_names = {'Normal', 'Sag', 'Swell', 'Interruption', ...
               'Harmonics', 'Transient', 'Flicker', 'Notching'};

labels = categorical(all_labels, 0:7, class_names);

% ---------------------------------------------------------
% SECTION 4: Save Dataset
% ---------------------------------------------------------
% Make sure the Data folder exists
if ~exist('../Data', 'dir')
    mkdir('../Data');
end

save('../Data/pq_dataset.mat', 'all_signals', 'labels', ...
     'all_labels', 'fs', 'f0', 'N', 'cycles', 'n_pts', ...
     'class_names', 'num_per_class', 'n_classes');

fprintf('Dataset saved to: ../Data/pq_dataset.mat\n\n');

% ---------------------------------------------------------
% SECTION 5: Plot All 8 Signal Classes
% ---------------------------------------------------------
% We pick the FIRST sample of each class and plot it
% so you can visually verify each waveform looks correct

fprintf('Generating waveform plots...\n');

% Indices of the first sample of each class
% Class 0 starts at row 1, Class 1 at row 2, etc.
% (because we interleaved them: 0,1,2,...,7,0,1,2,...,7,...)
first_idx = 1 : n_classes;   % rows 1 through 8

% Colours for each class plot
colours = {'#2196F3', '#FF5722', '#4CAF50', '#9C27B0', ...
           '#FF9800', '#F44336', '#009688', '#3F51B5'};

% Create a figure with 8 subplots (2 rows × 4 columns)
figure('Name', 'PQ Signal Waveforms — All 8 Classes', ...
       'NumberTitle', 'off', ...
       'Position', [100, 100, 1400, 700]);

for c = 1 : n_classes
    subplot(2, 4, c);
    
    sig = all_signals(first_idx(c), :);  % Get this class's signal
    t_ms = t * 1000;                      % Convert time to milliseconds
    
    plot(t_ms, sig, 'Color', colours{c}, 'LineWidth', 1.2);
    
    title(class_names{c}, 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Time (ms)', 'FontSize', 10);
    ylabel('Amplitude (pu)', 'FontSize', 10);
    ylim([-2.0, 2.0]);
    xlim([0, max(t_ms)]);
    grid on;
    grid minor;
    
    % Draw a horizontal reference line at y=0
    yline(0, 'k--', 'Alpha', 0.3);
end

sgtitle('Power Quality Event Waveforms — IEEE 1159 Synthetic Dataset', ...
        'FontSize', 15, 'FontWeight', 'bold');

% Save the figure
if ~exist('../Figures', 'dir')
    mkdir('../Figures');
end
saveas(gcf, '../Figures/Stage1_AllWaveforms.png');
fprintf('Figure saved to: ../Figures/Stage1_AllWaveforms.png\n');

% ---------------------------------------------------------
% SECTION 6: Plot Label Distribution (Bar Chart)
% ---------------------------------------------------------
figure('Name', 'Class Distribution', ...
       'NumberTitle', 'off', ...
       'Position', [100, 100, 800, 400]);

counts = histcounts(all_labels, 0:n_classes);
bar(0:n_classes-1, counts, 'FaceColor', '#1A4FA0', 'EdgeColor', 'none');
xticks(0:n_classes-1);
xticklabels(class_names);
xtickangle(30);
ylabel('Number of Samples', 'FontSize', 12);
title('Class Distribution — Balanced Dataset', ...
      'FontSize', 13, 'FontWeight', 'bold');
ylim([0, num_per_class * 1.3]);
grid on; grid minor;

% Add count labels on top of each bar
for i = 1:n_classes
    text(i-1, counts(i) + 10, num2str(counts(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 11, ...
         'FontWeight', 'bold');
end

saveas(gcf, '../Figures/Stage1_ClassDistribution.png');

fprintf('Class distribution plot saved.\n\n');

% ---------------------------------------------------------
% SECTION 7: Print Summary Statistics
% ---------------------------------------------------------
fprintf('==============================================\n');
fprintf('  STAGE 1 COMPLETE — Summary\n');
fprintf('==============================================\n');
fprintf('Total signals generated : %d\n', n_total);
fprintf('Signals per class       : %d\n', num_per_class);
fprintf('Signal length           : %d samples (%.3f sec)\n', ...
        n_pts, n_pts/fs);
fprintf('Sampling frequency      : %d Hz\n', fs);
fprintf('Classes:\n');
for c = 1:n_classes
    fprintf('  Class %d: %-15s — %d samples\n', ...
            c-1, class_names{c}, counts(c));
end
fprintf('\nReady for Stage 2: Preprocessing\n');
fprintf('==============================================\n');
