%% ����������б����������
aae.gArchitecture = [784 100];
aae.dArchitecture = [100 50 10 2];
aae.gEncoder = 'Sigmoid';
aae.gDecoder = 'Sigmoid';
aae.dActFunc = 'Sigmoid';

%% �Ż�����
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 50;
opt.numEpochs = 400;
opt.dStep = 100;
opt.gStep = 100;

%% ��ȡ����
[x, y, ~, ~] = loadMnist2();

%% ѵ������
aae = aaeTrain(aae, opt, x, y);