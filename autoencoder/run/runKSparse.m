%% ��������
clear all;
sae.visible = 784;
sae.hidden = 100;
sae.tied = 0;
sae.function = @aeKSparse;
sae.ksparse = 0.1;
sae.weightdecay = 0.00;
sae.encoder = 'Sigmoid';
sae.decoder = 'Linear';

%% �Ż�����
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.;
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ����
[x, y, ~, ~] = loadMnist2();

%% ѵ��
sae = saeTrain(sae, opt, x, y);

