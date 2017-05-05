%% 网络配置
ae.visible = 784;
ae.hidden = 100;
ae.tied = 1;
ae.encoder = 'Sigmoid';
ae.decoder = 'Sigmoid';

%% 优化参数
opt.momentum = 0.9;
opt.learnRate = 0.05;
opt.batchSize = 16;
opt.numEpochs = 400;

%% 数据
[x, y, ~, ~] = loadMnist2();

%% 训练
ae = aeTrain(ae, opt, x, y);

