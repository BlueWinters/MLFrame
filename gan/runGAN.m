%% ����������б����������
gan.genor = [32 100 784];
gan.disor = [784 100 10];

%% �Ż�����
opt.optMethod = @ganSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;
opt.kStep = 5;


%% ��ȡ����
data = loadMNIST();

%% ѵ������
gan = ganCheckConfig(gan, x, y);
gan = ganTrain(gan, opt, x, y);