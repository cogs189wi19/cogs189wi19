%P300_Preprocessing.m
% Purpose: Pre-process P300 competition data
% Data Description: http://www.bbci.de/competition/ii/albany_desc/albany_desc_ii.pdf
%
% The goal is to pre-process this data so that it can be exported to Python
% We will be using NumPy, so we're using a 3rd-party function to export
% our data to .npy format
%
% Created.: Alessandro "Ollie" D'Amico
% GitHub..: ollie-d
% Modified: 12 Feb 2019
%
% Known Bugs:
% - Train data is being exported with 1 extra sample. No idea why; low importance

clearvars
out_dir = 'C:\Users\H8801\Desktop\A2_data\export\';
% We want to bin the data from about 0ms to 500ms after response
% since by looking at waveform and r2 from the write-up, there isn't really
% any interesting data after 500ms.
% Let's define out srate
Fs = 240;        % Hz
dt = 1000/Fs;    % Ms
sdt = round(dt); % rounded ms
nchans = 64;     % number of channels

% Design filter
lp = 30;   % Hz
hp = 0.1;  % Hz
[b1,a1] = butter(2,lp/(Fs/2)); % lowpass filter
[b2,a2] = butter(2,hp/(Fs/2),'high'); % highpass filter

% Let's set some variables to epoch and baseline correct the data
ep_s = 0; % 0ms
ep_e = 500; % 500ms after stim
bl_s = 0; % 0ms
bl_e = 100; % 100ms after stim
p3_s = 250;
p3_e = 450;

% Create the index values using these times
i_ep_s = round(ep_s / sdt);
i_ep_e = round(ep_e / sdt);
i_p3_s = round(p3_s / sdt);
i_p3_e = round(p3_e / sdt);
i_bl_s = round(bl_s / sdt) + 1;
i_bl_e = round(bl_e / sdt) + 1;

% Define sub-samling rate for export
k = 10;
dim = (i_p3_e - i_p3_s) / k; % tot number of samples exported

% For saving aggregated data
train_df = zeros(dim, nchans, 8000);
train_la = zeros(8000, 2);
test_df = zeros(dim, nchans, 8000);
test_la = zeros(8000, 1);
train_count = 0;
test_count  = 0;
train_break = [];
test_break = [];

% Iterate through .mat files
pfx = 'AAS0';
ses = 10:12;
run = 1:8;
for s = ses
    for r = run
        % Try loading data, if not possible, continue
        filename = [pfx num2str(s) 'R0' num2str(r)];
        try
            load([filename '.mat'])
        catch e
            continue;
        end

        signal = filtfilt(b1, a1, signal);
        signal = filtfilt(b2, a2, signal);

        % We only want the sample where a new row/col is initially flashed
        % For this, we can use StimulusCode where the pattern is 0->not 0
        % We can do this by finding every location that is 0 and subtracting
		% these data from a right-shifted version of itself
        zeroes_ix = find(StimulusCode == 0); % save all zeroes
        events_ix = zeroes_ix(2:end) - zeroes_ix(1:end-1);
        events_ix = zeroes_ix(events_ix ~= 1) + 1;

        % We can now create a sample_size * chan * epoch array of our data
        epoch_df = zeros(abs(i_ep_s - i_ep_e), nchans, size(events_ix, 1));
        data_df = zeros(dim, nchans, size(events_ix, 1));

        % And now we can store those epoched data
        for i = 1:size(events_ix, 1)
            s_ix = events_ix(i) - i_ep_s;
            e_ix = events_ix(i) + i_ep_e - 1;
            epoch_df(:, :, i) = signal(s_ix:e_ix, :);

            % baseline correct
            epoch_df(:, :, i) = epoch_df(:, :, i) - mean(epoch_df(i_bl_s:i_bl_e, :, i), 1);
            
            % Save subsampled data
            data_df(:, :, i) = epoch_df(i_p3_s:k:(i_p3_e - 1), :, i);
        end

        % If S12 save to test, otherwise train
        if s == 12
            labels = StimulusCode(events_ix);
            for q = 1:size(epoch_df, 3)
                test_count = test_count + 1;
                test_df(:, :, test_count) = data_df(:, :, q);
                test_la(test_count) = labels(q);
            end
            test_break = [test_break test_count-1]; % subtract one for python 0-indexing
        else
            labels = [StimulusType(events_ix) StimulusCode(events_ix)];
            for q = 1:size(epoch_df, 3)
                train_count = train_count + 1;
                train_df(:, :, train_count) = data_df(:, :, q);
                train_la(train_count, :) = labels(q, :);
            end
            train_break = [train_break train_count-1]; % subtract one for python 0-indexing
        end
        
        % Export data
%         filename = [pfx num2str(s) 'R0' num2str(r)]; % something funky was happening with S12
%         writeNPY(epoch_df, [out_dir filename '_epoch.npy']);
%         csvwrite([out_dir filename '_label.csv'], labels);
    end
end

% Trim data
train_df = train_df(:, :, 1:train_count);
train_la = train_la(1:train_count, :);
test_df = test_df(:, :, 1:test_count);
test_la = test_la(1:test_count);

% Export data for numpy (https://github.com/kwikteam/npy-matlab)
writeNPY(train_df, [out_dir 'train_df.npy']);
csvwrite([out_dir 'train_la.csv'], train_la);
writeNPY(test_df, [out_dir 'test_df.npy']);
csvwrite([out_dir 'test_la.csv'], test_la);
%csvwrite([out_dir 'train_breaks.csv'], train_break);
%csvwrite([out_dir 'test_breaks.csv'], test_break);
