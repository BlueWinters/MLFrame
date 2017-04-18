%% ��������
saeCfg.architecture = [784 100];

%% �Ż�����
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.05;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ������������
saeCfg.size = size(saeCfg.architecture,2);
saeCfg.layerCfg = cell(saeCfg.size-1,1);

saeCfg.layerCfg{2}.tied = 0;
saeCfg.layerCfg{2}.function = @aeKSparse;
saeCfg.layerCfg{2}.ksparse = 0.2;
saeCfg.layerCfg{2}.weightdecay = 0.01;
saeCfg.layerCfg{2}.encoder = 'Sigmoid';
saeCfg.layerCfg{2}.decoder = 'Linear';

%% ����
data = loadMNIST();

%% ѵ��
sae = saeTrain(saeCfg, opt, data.xTrain, data.yTrain);

