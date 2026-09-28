function [] = ts_hctsa_workers(doPool,nworkers,inpLabel,featureLabel,inpMatpath,initMatpath,outFtspath,hctsaPath,mopsFile,opsFile)
% Compute the configured hctsa operations for every region in one input.
% Based on and modified from: https://github.com/linktianx/RSRD_BWAS

addpath(genpath(hctsaPath));

inpmatpath = fullfile(inpMatpath,strcat(inpLabel,'_inp.mat'));
data_inp = load(inpmatpath);
ts_data = double(data_inp.data);
[~,k] = size(ts_data);

initmat = fullfile(initMatpath,strcat(inpLabel,'_init.mat'));
labels = cell(1,k);
timeSeriesData = cell(1,k);
keywords = cell(1,k);

for i = 1:k
    timeSeriesData{1,i} = ts_data(:,i);
    labels{1,i} = data_inp.labels(i,:);
end
save(initmat,'timeSeriesData','labels','keywords');

outmatpath = fullfile(outFtspath,featureLabel,strcat(inpLabel,'_out.mat'));
TS_Init(initmat,mopsFile,opsFile,false,outmatpath);
TS_Compute(doPool,nworkers,[],[],[],outmatpath,false);
end
