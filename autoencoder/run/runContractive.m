%% ��������
cae.visible = 784;
cae.hidden = 100;
cae.tied = 1;
cae.encoder = 'Sigmoid';
cae.decoder = 'Sigmoid';
cae.lambda = 0.0001;
cae.weightdecay = 0.00;

%% �Ż�����
opt.momentum = 0.9;
opt.learnRate = 0.05;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ����
[x, y, ~, ~] = loadMnist2();

%% ѵ��
cae = caeTrain(cae, opt, x, y);

