%% ��������
clear all;
sae.visible = 784;
sae.hidden = 100;
sae.tied = 0;
sae.function = @saeSparse;
sae.encoder = 'Sigmoid';
sae.decoder = 'Linear';
sae.sparsity = 0.1;
sae.beta = 3;
sae.weightdecay = 0.01;

%% �Ż�����
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.1;
opt.batchSize = 40;
opt.numEpochs = 400;

%% ����
[x, y, ~, ~] = loadMnist2();

%% ѵ��
sae = saeTrain(sae, opt, x, y);

