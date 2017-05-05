%% 网络配置
cae.visible = 784;
cae.hidden = 100;
cae.tied = 1;
cae.encoder = 'Sigmoid';
cae.decoder = 'Sigmoid';
cae.lambda = 0.0001;
cae.weightdecay = 0.00;

%% 优化方法
opt.momentum = 0.9;
opt.learnRate = 0.05;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 数据
[x, y, ~, ~] = loadMnist2();

%% 训练
cae = caeTrain(cae, opt, x, y);

