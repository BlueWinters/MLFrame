function dae = daeTrain(dae, opt, x, y)

% ��ʱ����
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

% ����ͳ�Ƶı���
loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);
iter = 1;

% ����������Ŷ�
nx = daeMakeDenoise(dae, x);

% ��ʼ��Ȩֵ
dae = aeInitParameters(dae);
% ��ʼ���м����
mid = aeInitMidMt(dae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        nxBatch = nx(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % ��ʧ��������
        mid = daeDenoise(dae, mid, xBatch, nxBatch, yBatch);
        % �ݶ��½��Ż�����
        dae = aeSgdMomentum(dae, opt, mid);
        
        loss(iter) = mid.loss;
        iter = iter + 1;
    end

    time = toc;
    
    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
%     visual2(dae.w1',[10 10]);
    display_network(dae.w1');
    subplot(2,2,2);
%     visual2(dae.w2,[10 10]);
    display_network(dae.w2);
   
    subplot(2,2,3);
    plot([1:i],mloss(1:i), 'b');
    axis([0 numEpochs 0 ceil(max(mloss))]);
    
	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mloss(i)) '.']);
end

end

%%
function mid = aeInitMidMt(ae)
mid.vw1 = zeros(size(ae.w1));
mid.vw2 = zeros(size(ae.w2));
mid.vb1 = zeros(size(ae.b1));
mid.vb2 = zeros(size(ae.b2));
end