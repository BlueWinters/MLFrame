%% ��������
sgae.visible = 784;
sgae.hidden = 100;

sgae.tied = 0;
sgae.encoder = 'Sigmoid';
sgae.decoder = 'Linear';

sgae.function = @sgaeBasic;
% sgae.gmethod = 'average';
sgae.gsize = 20;
% sgae.giter = 0.05;
sgae.glambda = 0.003;
sgae.weightdecay = 0;

%% �Ż�����
opt.optMethod = @aeSgdMomentum;
opt.momentum = 0.9;
opt.learnRate = 0.05;
opt.batchSize = 20;
opt.numEpochs = 400;

%% ����
[x, y, ~, ~] = loadMnist2();

%% ѵ��
sgae = sgaeTrain(sgae, opt, x, y);

