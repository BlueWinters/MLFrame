%% ��������
ae.v = 784;
ae.h = 100;
ae.encoder = 'Sigmoid';
ae.decoder = 'Sigmoid';

%% �Ż�����
opt.rate = 1;
opt.method = @sgd;
opt.batchSize = 100;
opt.numEpochs = 400;

%% ����
data = loadMNIST();

%% ѵ��
% ae = aeTrain(ae, opt, data.xTrain, data.yTrain);
aeCheckGrad(ae);