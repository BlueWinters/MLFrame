function ae = caeTrain(ae, opt, x, y)

% 临时变量
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);
jacobi = zeros(numEpochs*numBatches, 1);
mjacobi = zeros(numEpochs, 1);

iter = 1;

% 初始化权值
ae = aeInitParameters(ae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % 损失函数计算
        mid = ae.function(ae, xBatch, yBatch);
        % 梯度下降优化参数
        ae = opt.optMethod(ae, opt, mid);
        
        loss(iter) = mid.loss;
        jacobi(iter) = mid.jacobi;
        iter = iter + 1;
    end

    time = toc;
    
    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    mjacobi(i) = mean(jacobi((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
    display_network(ae.w1');
    subplot(2,2,2);
    display_network(ae.w2);
   
    subplot(2,2,3);
    plot([1:i],mloss(1:i), 'b');
    axis([0 numEpochs 0 ceil(max(mloss))]);
    subplot(2,2,4);
    plot([1:i],mjacobi(1:i), 'b');    
    axis([0 numEpochs 0 ceil(max(mloss))]);
    
	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mloss(i)) '.']);
end

end