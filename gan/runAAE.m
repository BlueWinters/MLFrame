%% ����������б����������
aae.gArchitecture = [784 100];
aae.dArchitecture = [100 50 10 2];
aae.gEncoder = 'Sigmoid';
aae.gDecoder = 'Sigmoid';
aae.dActFunc = 'Sigmoid';

%% �Ż�����
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;
opt.dStep =10;
opt.gStep = 10;

%% ��ȡ����
[x, y] = loadMNIST();

%% ѵ������
aae = aaeTrain(aae, opt, x, y);