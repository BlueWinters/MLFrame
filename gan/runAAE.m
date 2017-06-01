%% 生成网络和判别网络的设置
aae.gArchitecture = [784 100];
aae.dArchitecture = [100 50 10 2];
aae.gEncoder = 'Sigmoid';
aae.gDecoder = 'Sigmoid';
aae.dActFunc = 'Sigmoid';

%% 优化参数
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 50;
opt.numEpochs = 400;
opt.dStep = 100;
opt.gStep = 100;

%% 读取数据
[x, y, ~, ~] = loadMnist2();

%% 训练网络
aae = aaeTrain(aae, opt, x, y);