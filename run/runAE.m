%% 网络配置
ae.v = 784;
ae.h = 100;
ae.encoder = 'Sigmoid';
ae.decoder = 'Sigmoid';

%% 优化方法
opt.rate = 1;
opt.method = @sgd;
opt.batchSize = 100;
opt.numEpochs = 400;

%% 数据
data = loadMNIST();

%% 训练
% ae = aeTrain(ae, opt, data.xTrain, data.yTrain);
aeCheckGrad(ae);