%% ��������
ae.visible = 784;
ae.hidden = 100;
ae.tied = 1;
ae.encoder = 'Sigmoid';
ae.decoder = 'Sigmoid';

%% �Ż�����
opt.momentum = 0.9;
opt.learnRate = 0.05;
opt.batchSize = 16;
opt.numEpochs = 400;

%% ����
[x, y, ~, ~] = loadMnist2();

%% ѵ��
ae = aeTrain(ae, opt, x, y);

