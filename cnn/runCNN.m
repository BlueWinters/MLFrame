%% 网络配置

cnn.layers = {
    struct('type', 'input', 'inputSize', [32 32], 'inputChls', 3)
    struct('type', 'conv', 'kernelSize', [5 5], 'kernelNums', 32)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'pool', 'scaleSize', [2 2], 'scaleType', 'Max') 
%     struct('type', 'conv', 'kernelSize', [5 5], 'kernelNums', 6)
%     struct('type', 'act', 'function', 'ReLU')
%     struct('type', 'pool', 'scaleSize', [2 2], 'scaleType', 'Max')
    struct('type', 'fc', 'layerSet', [100 50 20 10], 'function', 'Sigmoid')
};
cnn.size = numel(cnn.layers);

%% 优化方法
opt.learnRate = 10;
opt.optMethod = @cnnSgdMomentum;
opt.batchSize = 20;
opt.numEpochs = 400;

%% 数据
[x, y] = loadCifar10();

%% 训练
cnn = cnnCheckConfig(cnn, opt, x, y);
cnn = cnnTrain(cnn, opt, x, y);

