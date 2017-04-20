%% ��������

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

%% �Ż�����
opt.learnRate = 1;
opt.method = @cnnSgdMomentum;
opt.batchSize = 10;
opt.numEpochs = 400;

%% ����
[x, y] = loadMnist4();

%% ѵ��
cnn = cnnTrain(cnn, opt, x, y);

