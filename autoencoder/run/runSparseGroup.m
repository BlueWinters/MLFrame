%% ��������
sgae.v = 784;
sgae.h = 100;

sgae.tied = 0;
sgae.encoder = 'Sigmoid';
sgae.decoder = 'Sigmoid';

sgae.function = @sgaeBasic;
% sgae.gmethod = 'average';
sgae.gsize = 20;
% sgae.giter = 0.05;
sgae.glambda = 0.001;
sgae.weightdecay = 0.0001;

%% �Ż�����
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 10;
opt.numEpochs = 400;

%% ����
data = loadMNIST();

%% ѵ��
sae = sgaeTrain(sgae, opt, data.xTrain, data.yTrain);

