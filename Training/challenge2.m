



% biopil.backend.toolbox_manager('add', 'eeglab');

% releases = {'R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11'};
% tasks = {'contrastChangeDetection'};
% preprocessData(releases, tasks, 10, true);

% releases = {'R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11'};
% tasks = {'contrastChangeDetection'};
% preprocessData(releases, tasks, 10, false);

% releases = {'R5'};
% tasks = {'contrastChangeDetection'};
% preprocessData(releases, tasks, [], []);

% releases = {'NC'};
% tasks = {'contrastChangeDetection'};
% preprocessDataNC(releases, tasks, [], []);

%% Merge training + NC
mergeTrainingNC();

% toParquet('/Volumes/GREY SSD/cache/scratch/2s/challenge2', '/Volumes/GREY SSD/cache/scratch/2s/challenge2_parquet', 'R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training.mat');
toParquet('/Volumes/GREY SSD/cache/scratch/2s/challenge2', '/Volumes/GREY SSD/cache/scratch/2s/challenge2_parquet', 'R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training.mat');
% toParquet('/Volumes/GREY SSD/cache/scratch/2s/challenge2', '/Volumes/GREY SSD/cache/scratch/2s/challenge2_parquet', 'R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_validation.mat');
% toParquet('/Volumes/GREY SSD/cache/scratch/2s/challenge2', '/Volumes/GREY SSD/cache/scratch/2s/challenge2_parquet', 'R5_contrastChangeDetection.mat');
% toParquet('/Volumes/GREY SSD/cache/scratch/2s/challenge2', '/Volumes/GREY SSD/cache/scratch/2s/challenge2_parquet', 'NC_contrastChangeDetection.mat');

return;

function mergeTrainingNC()
    training = load('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_9.mat');
    trainingMeta = load('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training.mat');

    nc = load('/Volumes/GREY SSD/cache/scratch/2s/challenge2/NC_contrastChangeDetection_1.mat');
    ncMeta = load('/Volumes/GREY SSD/cache/scratch/2s/challenge2/NC_contrastChangeDetection.mat');

    combined = struct;
    combined.labels = cat(1, training.labels, nc.labels);
    combined.batches = cat(1, training.batches, nc.batches);

    combinedMeta = trainingMeta;
    combinedMeta.cacheName = 'R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training';
    combinedMeta.nbBatches = trainingMeta.nbBatches + ncMeta.nbBatches;

    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_1.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_1.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_2.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_2.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_3.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_3.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_4.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_4.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_5.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_5.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_6.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_6.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_7.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_7.mat');
    copyfile('/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_contrastChangeDetection_training_8.mat', '/Volumes/GREY SSD/cache/scratch/2s/challenge2/R1_R2_R3_R4_R6_R7_R8_R9_R10_R11_NC_contrastChangeDetection_training_8.mat');
    save(fullfile(combinedMeta.cacheFolder, strcat(combinedMeta.cacheName, '.mat')), '-struct', 'combinedMeta');
    save(fullfile(combinedMeta.cacheFolder, strcat(combinedMeta.cacheName, '_9.mat')), '-struct', 'combined');
end


function data = highPassFilter(data, cutoff, fs)
    [z, p, k] = butter(6, cutoff/(fs/2), 'high');
    [sos, g] = zp2sos(z, p, k);
    data = filtfilt(sos, g, data);
end

function data = lowPassFilter(data, cutoff, fs)
    [z, p, k] = butter(6, cutoff/(fs/2), 'low');
    [sos, g] = zp2sos(z, p, k);
    data = filtfilt(sos, g, data);
end

function tasks = listTasks(folder, release, participant)
    filepath = fullfile(folder, sprintf('%s', release{1}), participant, 'eeg', sprintf('%s_task-*_eeg.bdf', participant));
    files = dir(filepath);
    tasks = arrayfun(@(x)(transpose(strsplit(strrep(strrep(x.name, sprintf('%s_task-', participant), ''), '_eeg.bdf', ''), '_'))), files, 'UniformOutput', false);
    tasks = cellfun(@(x)(x{1}), tasks, 'UniformOutput', false);
    tasks = unique(tasks);
end

function tasks = listTasksNC(folder, release, participant)
    filepath = fullfile(folder, sprintf('cmi_bids_%s', release{1}), participant, 'eeg', sprintf('%s_task-*_eeg.set', participant));
    files = dir(filepath);
    tasks = arrayfun(@(x)(transpose(strsplit(strrep(strrep(x.name, sprintf('%s_task-', participant), ''), '_eeg.set', ''), '_'))), files, 'UniformOutput', false);
    tasks = cellfun(@(x)(x{1}), tasks, 'UniformOutput', false);
    tasks = unique(tasks);
end

function data = loadbdf(behavioural, folder, release, participant, task, run)
    filepath = fullfile(folder, sprintf('%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_run-%s_eeg.bdf', participant, task, run));

    if ~exist(filepath, 'file')
        filepath = fullfile(folder, sprintf('%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_eeg.bdf', participant, task));
    end
    % disp(filepath);
    d = pop_biosig(filepath);

    assert(d.srate == 100);

    data.fs = d.srate;
    data.channels.number = d.nbchan;
    data.channels.locations = d.chanlocs;
    data.channels.info = d.chaninfo;
    data.ref = d.ref;
    data.events = d.urevent;
    data.eeg = d.data';
    data.participant = participant;
    data.task = task;
    data.run = run;
    data.release = release;
    data.behavioural = table2struct(behavioural(strcmp(behavioural.participant_id, sprintf('sub-%s', participant)), :));


    eventFilepath = fullfile(folder, sprintf('%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_run-%s_events.tsv', participant, task, run));
    if ~exist(eventFilepath, 'file')
        eventFilepath = fullfile(folder, sprintf('%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_events.tsv', participant, task));
    end
    data.events = readtable(eventFilepath, 'FileType','text');
end

function data = loadbdfNC(behavioural, folder, release, participant, task, run)
    filepath = fullfile(folder, sprintf('cmi_bids_%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_run-%s_eeg.set', participant, task, run));

    if ~exist(filepath, 'file')
        filepath = fullfile(folder, sprintf('cmi_bids_%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_eeg.set', participant, task));
    end
    % disp(filepath);
    d = pop_loadset(filepath);

    assert(d.srate == 500);
    fs = 100;

    eeg = d.data;
    eeg(end, :) = []; % remove Cz
    eeg = transpose(eeg); % we want (time x channels)
    % eeg = eeg - mean(eeg, 2); % CAR
    eeg = highPassFilter(eeg, 0.5, d.srate); % remove drift
    eeg = resample(eeg, fs, d.srate);

    data.fs = fs;
    data.channels.number = d.nbchan - 1;
    data.channels.locations = d.chanlocs;
    data.channels.info = d.chaninfo;
    data.ref = d.ref;
    data.events = d.urevent;
    data.eeg = eeg;
    data.participant = participant;
    data.task = task;
    data.run = run;
    data.release = release;
    data.behavioural = table2struct(behavioural(strcmp(behavioural.participant_id, sprintf('sub-%s', participant)), :));


    eventFilepath = fullfile(folder, sprintf('cmi_bids_%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_run-%s_events.tsv', participant, task, run));
    if ~exist(eventFilepath, 'file')
        eventFilepath = fullfile(folder, sprintf('cmi_bids_%s', release{1}), participant, 'eeg', sprintf('%s_task-%s_events.tsv', participant, task));
    end
    data.events = readtable(eventFilepath, 'FileType','text');
end

function x = normalise(x)
    x = squeeze(x);
    assert(isvector(x));
    x = x - mean(x);
    s = std(x);
    if s > 1e-6
        x = x / s;
    end
end

function preprocessData(releases, tasks, participantSplit, training)
    % interestingChannels = [127, 126, 49, 113, 68, 94, 11, 62, 40, 109];
    interestingChannels = 1:128;
    nbChannels = length(interestingChannels);
    cacheName = '';
    for releaseIdx = 1:length(releases)
        if releaseIdx == 1
            cacheName = sprintf('%s', releases{releaseIdx});
        else
            cacheName = sprintf('%s_%s', cacheName, releases{releaseIdx});
        end
    end
    for taskIdx = 1:length(tasks)
        cacheName = sprintf('%s_%s', cacheName, tasks{taskIdx});
    end
    if ~isempty(participantSplit)
        if training
            cacheName = sprintf('%s_training', cacheName);
        else
            cacheName = sprintf('%s_validation', cacheName);
        end
    end

    folder = '/Volumes/RED SSD/NeurIPS';
    runs = {'1', '2', '3'};

    behavioural = loadExternalization(folder, releases);

    fs = 100;
    nbBatches = 0;
    cacheFolder = '/Volumes/GREY SSD/cache/scratch/2s/challenge2';
    cacheNumber = 1;
    batch_size = 2*fs;
    maxNbBatches = 50000; % caches will be around 5 GB
    batches = nan(maxNbBatches, nbChannels, batch_size, 'single');
    labels = nan(maxNbBatches, 1);
    batchIdx = 1;
    for rowIdx = 1:height(behavioural)
        fprintf('Subject %d/%d\n', rowIdx, height(behavioural));

        if ~isempty(participantSplit)
            if (mod(rowIdx, participantSplit) ~= 0) ~= training
                continue;
            end
        end

        availableTasks = listTasks(folder, behavioural{rowIdx, "release"}, behavioural.participant_id{rowIdx});

        for taskIdx = 1:length(availableTasks)
            availableTask = availableTasks{taskIdx};

            if ismember(availableTask, tasks)
                for runIdx = 1:length(runs)
                    run = runs{runIdx};
                    try
                        data = loadbdf(behavioural, folder, behavioural{rowIdx, "release"}, behavioural.participant_id{rowIdx}, availableTask, run);
                    catch exception
                        continue;
                    end

                    % Preprocessing
                    eeg = data.eeg;
                    if size(eeg, 2) ~= 128
                        continue;
                    end
                    eeg = eeg(:, interestingChannels);
                    % eeg = highPassFilter(eeg, highpassCutoff, data.fs);
                    % eeg = resample(eeg, targetFs, data.fs);
                    events = data.events;
                    samples = nan(height(events), 1);
                    for eventIdx = 1:height(events)
                        samples(eventIdx) = floor(events(eventIdx, :).onset*fs)+1;
                    end
                    events.sample = samples;

                    % Artefact suppression
                    % factor = prctile(eeg(:), 99);
                    % eeg = tanh(eeg/factor);

                    % Check if the EEG is long enough
                    if height(events) < 3
                        continue;
                    end

                    start = events{2, "sample"};
                    fin = events{end-2, "sample"};

                    % Cut EEG
                    eeg = eeg(start:fin, :);

                    start_batch = 1;
                    end_batch = start_batch + batch_size - 1;

                    while end_batch <= size(eeg, 1)
                        if batchIdx > size(batches, 1)
                            nbBatches = nbBatches + saveBatches(batches, labels, batchIdx, cacheFolder, cacheName, cacheNumber);
                            batchIdx = 1;
                            batches = nan(maxNbBatches, nbChannels, batch_size, 'single');
                            labels = nan(maxNbBatches, 1);
                            cacheNumber = cacheNumber + 1;
                        end
                        batches(batchIdx, :, :) = transpose(eeg(start_batch:end_batch, :));
                        labels(batchIdx) = behavioural{rowIdx, "externalizing"};

                        batchIdx = batchIdx + 1;
                        start_batch = end_batch + 1;
                        end_batch = start_batch + batch_size - 1;
                    end
                end
            end
        end
    end

    nbBatches = nbBatches + saveBatches(batches, labels, batchIdx, cacheFolder, cacheName, cacheNumber);
    save(fullfile(cacheFolder, sprintf('%s.mat', cacheName)), 'cacheFolder', 'cacheName', 'cacheNumber', 'nbBatches', 'fs', 'nbChannels');
end

function preprocessDataNC(releases, tasks, participantSplit, training)
    % interestingChannels = [127, 126, 49, 113, 68, 94, 11, 62, 40, 109];
    interestingChannels = 1:128;
    nbChannels = length(interestingChannels);
    cacheName = '';
    for releaseIdx = 1:length(releases)
        if releaseIdx == 1
            cacheName = sprintf('%s', releases{releaseIdx});
        else
            cacheName = sprintf('%s_%s', cacheName, releases{releaseIdx});
        end
    end
    for taskIdx = 1:length(tasks)
        cacheName = sprintf('%s_%s', cacheName, tasks{taskIdx});
    end
    if ~isempty(participantSplit)
        if training
            cacheName = sprintf('%s_training', cacheName);
        else
            cacheName = sprintf('%s_validation', cacheName);
        end
    end

    folder = '/Volumes/GREY SSD/NeurIPS';
    runs = {'1', '2', '3'};

    behavioural = loadExternalizationNC(folder, releases);

    fs = 100;
    nbBatches = 0;
    cacheFolder = '/Volumes/GREY SSD/cache/scratch/2s/challenge2';
    cacheNumber = 1;
    batch_size = 2*fs;
    maxNbBatches = 90000; % caches will be around 9 GB
    batches = nan(maxNbBatches, nbChannels, batch_size, 'single');
    labels = nan(maxNbBatches, 1);
    batchIdx = 1;
    for rowIdx = 1:height(behavioural)
        fprintf('Subject %d/%d\n', rowIdx, height(behavioural));

        if ~isempty(participantSplit)
            if (mod(rowIdx, participantSplit) ~= 0) ~= training
                continue;
            end
        end

        availableTasks = listTasksNC(folder, behavioural{rowIdx, "release"}, behavioural.participant_id{rowIdx});

        for taskIdx = 1:length(availableTasks)
            availableTask = availableTasks{taskIdx};

            if ismember(availableTask, tasks)
                for runIdx = 1:length(runs)
                    run = runs{runIdx};
                    try
                        data = loadbdfNC(behavioural, folder, behavioural{rowIdx, "release"}, behavioural.participant_id{rowIdx}, availableTask, run);
                    catch exception
                        continue;
                    end

                    % Preprocessing
                    eeg = data.eeg;
                    if size(eeg, 2) ~= 128
                        continue;
                    end
                    eeg = eeg(:, interestingChannels);
                    % eeg = highPassFilter(eeg, highpassCutoff, data.fs);
                    % eeg = resample(eeg, targetFs, data.fs);
                    events = data.events;
                    samples = nan(height(events), 1);
                    for eventIdx = 1:height(events)
                        samples(eventIdx) = floor(events(eventIdx, :).onset*fs)+1;
                    end
                    events.sample = samples;

                    % Artefact suppression
                    % factor = prctile(eeg(:), 99);
                    % eeg = tanh(eeg/factor);

                    % Check if the EEG is long enough
                    if height(events) < 3
                        continue;
                    end

                    start = events{2, "sample"};
                    fin = events{end-2, "sample"};

                    % Cut EEG
                    eeg = eeg(start:fin, :);

                    start_batch = 1;
                    end_batch = start_batch + batch_size - 1;

                    while end_batch <= size(eeg, 1)
                        if batchIdx > size(batches, 1)
                            nbBatches = nbBatches + saveBatches(batches, labels, batchIdx, cacheFolder, cacheName, cacheNumber);
                            batchIdx = 1;
                            batches = nan(maxNbBatches, nbChannels, batch_size, 'single');
                            labels = nan(maxNbBatches, 1);
                            cacheNumber = cacheNumber + 1;
                        end
                        batches(batchIdx, :, :) = transpose(eeg(start_batch:end_batch, :));
                        labels(batchIdx) = behavioural{rowIdx, "externalizing"};

                        batchIdx = batchIdx + 1;
                        start_batch = end_batch + 1;
                        end_batch = start_batch + batch_size - 1;
                    end
                end
            end
        end
    end

    nbBatches = nbBatches + saveBatches(batches, labels, batchIdx, cacheFolder, cacheName, cacheNumber);
    save(fullfile(cacheFolder, sprintf('%s.mat', cacheName)), 'cacheFolder', 'cacheName', 'cacheNumber', 'nbBatches', 'fs', 'nbChannels');
end

function nbBatches = saveBatches(batches, labels, batchIdx, cacheFolder, cacheName, cacheNumber)
    batches(batchIdx:end, :, :) = [];
    labels(batchIdx:end) = [];

    for batchIdx = 1:size(batches, 1)
        for channelIdx = 1:size(batches, 2)
            batches(batchIdx, channelIdx, :) = normalise(batches(batchIdx, channelIdx, :));
        end
    end

    save(fullfile(cacheFolder, sprintf('%s_%d.mat', cacheName, cacheNumber)), 'batches', 'labels');
    nbBatches = size(batches, 1);
end

function toParquet(loadCacheFolder, saveCacheFolder, cacheName)
    meta = load(fullfile(loadCacheFolder, cacheName));
    meta.cacheFolder = saveCacheFolder;

    nbBatches = 0;
    for idx = 1:meta.cacheNumber
        fprintf('Cache %d of %d\n', idx, meta.cacheNumber);
        d = load(fullfile(loadCacheFolder, sprintf('%s_%d.mat', meta.cacheName, idx)));
        
        removeMask = isnan(d.labels);
        d.labels(removeMask) = [];
        d.batches(removeMask, :, :) = [];
        nbBatches = nbBatches + length(d.labels);

        nbChannels = size(d.batches, 2);
        nbRows = size(d.batches, 1) * size(d.batches, 2);
        nbColumns = 3 + size(d.batches, 3);
        vartypes = repmat({'single'}, 1, nbColumns);
        t = table('Size', [nbRows, nbColumns], 'VariableTypes', vartypes);
        for batchIdx = 1:size(d.batches, 1)
            if mod(batchIdx, 1000) == 1
                fprintf('\tBatch %d of %d\n', batchIdx, size(d.batches, 1));
            end

            rowRange = (batchIdx-1)*nbChannels+1:batchIdx*nbChannels;
            t{rowRange, 1} = batchIdx;
            t{rowRange, 2} = transpose(1:nbChannels);
            t{rowRange, 3} = d.labels(batchIdx);
            t{rowRange, 4:end} = squeeze(d.batches(batchIdx, :, :));
        end
        parquetwrite(fullfile(saveCacheFolder, sprintf('%s_%d.parquet', meta.cacheName, idx)), t);
    end

    meta.nbChannels = nbChannels;
    meta.nbBatches = nbBatches;
    save(fullfile(saveCacheFolder, cacheName), '-struct', 'meta');
end

function behavioural = loadExternalizationNC(folder, releases)
    behavioural = table;
    for releaseIdx = 1:length(releases)
        currentBehavioural = readtable(fullfile(folder, sprintf('cmi_bids_%s', releases{releaseIdx}), 'participants.tsv'), 'FileType', 'text');
        currentBehavioural = currentBehavioural(:, {'participant_id', 'externalizing', 'release_number'});
        currentBehavioural.release_number = repmat({'NC'}, height(currentBehavioural), 1);
        behavioural = cat(1, behavioural, currentBehavioural);
    end
    behavioural.release = behavioural.release_number;
    behavioural.release_number = [];
end

function behavioural = loadExternalization(folder, releases)
    behavioural = table;
    for releaseIdx = 1:length(releases)
        currentBehavioural = readtable(fullfile(folder, sprintf('%s', releases{releaseIdx}), 'participants.tsv'), 'FileType', 'text');
        currentBehavioural = currentBehavioural(:, {'participant_id', 'externalizing', 'release_number'});
        behavioural = cat(1, behavioural, currentBehavioural);
    end
    behavioural.release = behavioural.release_number;
    behavioural.release_number = [];
end