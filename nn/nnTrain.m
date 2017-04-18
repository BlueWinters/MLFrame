function [nn]  = nnTrain(nn, opt, x, y)

% 临时变量
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);
iter = 1;

% 初始化网络权值
nn = nnInitialize(nn);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        mid = nnFeedforward(nn, xBatch, yBatch);
        mid = nnBackpropagate(nn, mid);
        nn = nnUpdateGrads(nn, opt, mid);
        
        loss(iter) = mid.loss;
        iter = iter + 1;
    end

    time = toc;

    mloss(i) = mean(loss((iter-numBatches):(iter-1)));

    subplot(1,2,1);
    display_network(nn.w{1}');
    subplot(1,2,2);
    plot([1:i], mloss(1:i));
    axis([0 numEpochs 0 ceil(max(mloss))]);
    
    
	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mean(loss((iter-numBatches):(iter-1)))) '.']);
end


end