%% 生成网络和判别网络的设置
gan.genor = [32 100 784];
gan.disor = [784 100 10];

%% 优化参数
opt.optMethod = @ganSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;
opt.kStep = 5;


%% 读取数据
data = loadMNIST();

%% 训练网络
gan = ganCheckConfig(gan, x, y);
gan = ganTrain(gan, opt, x, y);