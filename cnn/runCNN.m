%% 网络配置

cnn.layers = {
    struct('type', 'input', 'inputSize', [28 28], 'inputChls', 1)
    struct('type', 'conv', 'kernelSize', [5 5], 'kernelNums', 10)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'pool', 'scaleSize', [2 2], 'scaleType', 'Max') 
%     struct('type', 'conv', 'kernelSize', [5 5], 'kernelNums', 6)
%     struct('type', 'act', 'function', 'ReLU')
%     struct('type', 'pool', 'scaleSize', [2 2], 'scaleType', 'Max')
    struct('type', 'fc', 'layerSet', [100 50 20 10], 'function', 'Sigmoid')
};
cnn.size = numel(cnn.layers);

%% 优化方法
opt.learnRate = 1;
opt.method = @cnnSgdMomentum;
opt.batchSize = 10;
opt.numEpochs = 400;

%% 数据
[x, y] = loadMnist4();

%% 训练
cnn = cnnTrain(cnn, opt, x, y);

