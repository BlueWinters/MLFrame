function dae = daeTrain(dae, opt, x, y)

% ��ʱ����
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

% ����ͳ�Ƶı���
loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);

% ����������Ŷ�
nx = daeMakeDenoise(dae, x);

iter = 1;

% ��ʼ��Ȩֵ
dae = aeInitParameters(dae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        nxBatch = nx(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % ��ʧ��������
        mid = dae.function(dae, xBatch, nxBatch, yBatch);
        % �ݶ��½��Ż�����
        dae = opt.optMethod(dae, opt, mid);
        
        loss(iter) = mid.loss;
        iter = iter + 1;
    end

    time = toc;
    
    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
    display_network(dae.w1');
    subplot(2,2,2);
    display_network(dae.w2);
   
    subplot(2,2,3);
    plot([1:i],mloss(1:i), 'b');
    axis([0 numEpochs 0 ceil(max(mloss))]);
    
	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mloss(i)) '.']);
end

end