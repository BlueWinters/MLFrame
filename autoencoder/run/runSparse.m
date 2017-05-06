%% 网络配置
clear all;
sae.visible = 784;
sae.hidden = 100;
sae.tied = 0;
sae.function = @saeSparse;
sae.encoder = 'Sigmoid';
sae.decoder = 'Linear';
sae.sparsity = 0.1;
sae.beta = 3;
sae.weightdecay = 0.01;

%% 优化方法
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 40;
opt.numEpochs = 400;

%% 数据
[x, y, ~, ~] = loadMnist2();

%% 训练
sae = saeTrain(sae, opt, x, y);

