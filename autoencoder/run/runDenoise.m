%% ��������
dae.visible = 784;
dae.hidden = 100;
dae.tied = 1;
dae.noise = 'binary';
dae.fraction = 0.5;
dae.weightdecay = 0;
dae.encoder = 'Sigmoid';
dae.decoder = 'Sigmoid';

%% �Ż�����
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ����
[x, y, ~, ~] = loadMnist2();

%% ѵ��
dae = daeTrain(dae, opt, x, y);


