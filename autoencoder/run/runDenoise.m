%% 网络配置
dae.visible = 784;
dae.hidden = 100;
dae.tied = 1;
dae.noise = 'binary';
dae.fraction = 0.5;
dae.weightdecay = 0;
dae.encoder = 'Sigmoid';
dae.decoder = 'Sigmoid';

%% 优化方法
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 数据
[x, y, ~, ~] = loadMnist2();

%% 训练
dae = daeTrain(dae, opt, x, y);


