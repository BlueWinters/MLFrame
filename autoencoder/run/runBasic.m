%% 网络配置
saeCfg.architecture = [784 100];

%% 优化方法
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 网络逐层的配置
saeCfg.size = size(saeCfg.architecture,2);
saeCfg.layerCfg = cell(saeCfg.size-1,1);

saeCfg.layerCfg{2}.tied = 0;
saeCfg.layerCfg{2}.function = @aeBasic;
saeCfg.layerCfg{2}.encoder = 'Sigmoid';
saeCfg.layerCfg{2}.decoder = 'Linear';

%% 数据
data = loadMNIST();

%% 训练
sae = saeTrain(saeCfg, opt, data.xTrain, data.yTrain);

