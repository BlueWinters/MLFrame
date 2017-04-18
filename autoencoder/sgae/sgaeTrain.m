function sgae = sgaeTrain(sgae, opt, x, y)

% ��ʱ����
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);
gloss = zeros(numEpochs*numBatches, 1);
mgloss = zeros(numEpochs, 1);

iter = 1;

% ��ʼ��Ȩֵ
sgae = aeInitParameters(sgae);
% ��ʼ������Ϣ
group = sgaeInitGroups(sgae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % ��ʧ��������
        mid = sgae.function(sgae, group, xBatch, yBatch);
        % �ݶ��½��Ż�����
        sgae = opt.optMethod(sgae, opt, mid);
        
        loss(iter) = mid.loss;
        gloss(iter) = mid.gloss;
        iter = iter + 1;
    end

    time = toc;
    
    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    mgloss(i) = mean(gloss((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
    display_network(sgae.w1');
    subplot(2,2,2);
    display_network(sgae.w2);
    
    subplot(2,2,3);
    plot([1:i],mloss(1:i));
    axis([0 numEpochs 0 ceil(max(mloss))]);
    subplot(2,2,4);
    plot([1:i],mgloss(1:i));
    axis([0 numEpochs 0 ceil(max(mgloss))]);
    
	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mloss(i)) '. ' ...
        'group L2,1 ' num2str(mgloss(i)) '.']);
end

end