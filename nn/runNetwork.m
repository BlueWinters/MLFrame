%% 网络配置
nnConfig.architecture = [784 100 50 10];
nnConfig.size = size(nnConfig.architecture, 2);
nnConfig.layerCfg = cell(nnConfig.size-1,1);

%% 优化方法
opt.optMethod = @nnSgdMomentum;
opt.momentum = 0.9; 
opt.learnRate = 0.1;
opt.batchSize = 10;
opt.numEpochs = 400;

%% 数据
data = loadMNIST();

%% 网络逐层的配置
idx = 2;
nnConfig.layerCfg{idx}.actFunc = 'Sigmoid';
nnConfig.layerCfg{idx}.dropout = 0;
nnConfig.layerCfg{idx}.function = nnCfgPenalty('weightL2', @weightL2, 0.01);

idx = 3;
nnConfig.layerCfg{idx}.actFunc = 'Sigmoid';
nnConfig.layerCfg{idx}.dropout = 0.;

idx = 4;
nnConfig.layerCfg{idx}.actFunc = 'Sigmoid';
nnConfig.layerCfg{idx}.dropout = 0.;

%% 训练
% nnConfig = nnTrain(nnConfig, opt, data.xTrain, data.yTrain);

