function dae = daeTrain(dae, opt, x, y)

% 临时变量
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

% 用于统计的变量
loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);

% 对输入添加扰动
nx = daeMakeDenoise(dae, x);

iter = 1;

% 初始化权值
dae = aeInitParameters(dae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        nxBatch = nx(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % 损失函数计算
        mid = dae.function(dae, xBatch, nxBatch, yBatch);
        % 梯度下降优化参数
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