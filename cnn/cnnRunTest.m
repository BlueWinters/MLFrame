%% ��������
clear;
cnn.layers = {
    struct('type', 'input', 'inputSize', [28 28], 'inputChls', 1)
    struct('type', 'conv', 'kernelSize', [3 3], 'kernelNums', 9)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'conv', 'kernelSize', [3 3], 'kernelNums', 9)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'pool', 'scaleSize', [2 2], 'scaleType', 'Max')
    struct('type', 'conv', 'kernelSize', [5 5], 'kernelNums', 9)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'fc', 'layerSet', [100 10], 'function', 'Sigmoid')
};
cnn.size = numel(cnn.layers);

%% �Ż�����
opt.learnRate = 0.2;
opt.optMethod = @cnnSgdMomentum;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ����
[x, y, tx, ty] = loadMnist4();

%% ѵ��
% cnn = cnnCheckConfig(cnn, opt, x, y);
cnn = cnnTrain(cnn, opt, x, y);

%% ����
acc = cnnPredict(cnn, tx, ty);
