%% 网络配置
clear;
cnn.layers = {
    struct('type', 'input', 'inputSize', [28 28], 'inputChls', 1)
    struct('type', 'conv', 'kernelSize', [11 11], 'kernelNums', 1)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'sparse', 'kSparse', 0.5)
    struct('type', 'fc', 'layerSet', [100 10], 'function', 'Sigmoid')
};
cnn.size = numel(cnn.layers);

%% 优化方法
opt.learnRate = 0.2;
opt.optMethod = @cnnSgdMomentum;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 数据
[x, y, tx, ty] = loadMnist4();

%% 训练
% cnn = cnnCheckConfig(cnn, opt, x, y);
cnn = cnnTrain(cnn, opt, x, y);

%% 测试
acc = cnnPredict(cnn, tx, ty);
