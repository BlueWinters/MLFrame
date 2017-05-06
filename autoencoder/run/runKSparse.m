%% 网络配置
clear all;
sae.visible = 784;
sae.hidden = 100;
sae.tied = 0;
sae.function = @aeKSparse;
sae.ksparse = 0.1;
sae.weightdecay = 0.00;
sae.encoder = 'Sigmoid';
sae.decoder = 'Linear';

%% 优化方法
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 数据
[x, y, ~, ~] = loadMnist2();

%% 训练
sae = saeTrain(sae, opt, x, y);

