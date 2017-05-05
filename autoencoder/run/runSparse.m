%% 网络配置
ae.visible = 784;
ae.hidden = 100;
ae.tied = 0;
ae.function = @saeSparse;
ae.encoder = 'Sigmoid';
ae.decoder = 'Sigmoid';
ae.sparsity = 0.3;
ae.beta = 3;
ae.weightdecay = 0.01;

%% 优化方法
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 数据
[x, y, ~, ~] = loadMnist2();

%% 训练
ae = saeTrain(ae, opt, x, y);

