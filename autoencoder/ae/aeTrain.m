function ae = aeTrain(ae, opt, x, y)

% ��ʱ����
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);
iter = 1;

% ��ʼ��Ȩֵ
ae = aeInitParameters(ae);
% ��ʼ���м����
mid = adInitMidMt(ae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % ��ʧ��������
        mid = aeBasic(ae, mid, xBatch, yBatch);
        % �ݶ��½��Ż�����
        [ae, mid] = aeSgdMomentum(ae, opt, mid);
        
        loss(iter) = mid.loss;
        iter = iter + 1;
    end

    time = toc;
    
    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
    display_network(ae.w1');
    subplot(2,2,2);
    display_network(ae.w2);
    subplot(2,2,3);
    plot([1:i],mloss(1:i));
    axis([0 numEpochs 0 ceil(max(mloss))]);
    
	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mloss(i)) '.']);
end

end

%%
function mid = adInitMidMt(ae)
mid.vw1 = zeros(size(ae.w1));
mid.vw2 = zeros(size(ae.w2));
mid.vb1 = zeros(size(ae.b1));
mid.vb2 = zeros(size(ae.b2));
end