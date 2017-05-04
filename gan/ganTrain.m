function gan = ganTrain(gan, opt, x, y)

% 临时变量
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
kStep = opt.kStep;

% 初始化权值
gan = ganInitParameters(gan);

% 训练
for i = 1 : numEpochs
    tic;
    
    for k = 1 : kStep
        % 采样
        zBatch = 0;
        xBatch = 0;
        yBatch = 0;
        
        % 判别网络的损失函数计算和梯度更新
        dmid = ganDiscriminatorCost(gan, xBatch, yBatch, zBatch);
        gan = ganDiscriminatorUpdate(gan, opt, dmid);
    end
    
    % 采样
    z = 0;
    % 生成网络的损失函数计算和梯度更新
    gmid = ganGeneratorCost(gan, z);
    gan = ganGeneratorUpdate(gan, opt, gmid);
end


end