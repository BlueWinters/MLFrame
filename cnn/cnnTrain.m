function [cnn]  = cnnTrain(cnn, opt, x, y)

% ¡Ÿ ±±‰¡ø
nCases = size(x, 1);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = nCases / batchSize;

cnn = cnnInitialize(cnn, x);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, :, :, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        mid = cnnFeedforward(cnn, xBatch, yBatch);
        mid = cnnBackpropagate(cnn, mid);
        cnn = cnnUpdateGrads(cnn, mid);
        loss
    end

    time = toc;
    
	disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '.'...
        'Took ' num2str(time) ' seconds.' ...
        'Mini-batch mean squared error ' num2str(mean(L((n-numBatches):(n-1)))) '.']);
%     network.learningRate = network.learningRate * network.scaling_learningRate;
end