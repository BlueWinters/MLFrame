%% ��������

cnn.layers = {
    struct('type', 'input')
    struct('type', 'conv', 'kernelSize', 5, 'kernelNums', 6)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'pool', 'scaleSize', 2, 'scaleType', 'Mean') 
    struct('type', 'conv', 'kernelSize', 5, 'kernelNums', 6)
    struct('type', 'act', 'function', 'ReLU')
    struct('type', 'pool', 'scaleSize', 2, 'scaleType', 'Mean')
    struct('type', 'fc', 'layerSet', [100 50 20 10], 'function', 'Sigmoid')
};
cnn.size = numel(cnn.layers);

%% �Ż�����
opt.learnRate = 0.1;
opt.method = @cnnSgdMomentum;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ����
[x, y] = loadMnist4();

%% ѵ��
cnn = cnnTrain(cnn, opt, x, y);

