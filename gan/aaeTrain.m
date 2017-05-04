function aae = aaeTrain(aae, opt, x, y)

% 临时变量
nCases = size(x,2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
dStep = opt.kStep;
gStep = opt.gStep;
zSize = aae.dArchitecture(1);

dloss = zeros(numEpochs*dStep,1);
gloss = zeros(numEpochs*gStep,2);
diter = 0;
giter = 0;

% 初始化权值
aae = aaeInitParameters(aae);
[gmid, dmid] = aaeInitMidMt(aae);

% 训练
for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for k = 1 : dStep
        % 采样
        zBatch = rand(zSize, batchSize);
        xBatch = x(:, index((k - 1) * batchSize + 1 : k * batchSize));
        
        % 判别网络的损失函数计算和梯度更新
        dmid = aaeDiscriminatorCost(aae, dmid, xBatch, zBatch);
        aae = aaeDiscriminatorUpdate(aae, opt, dmid);
        
        diter = diter + 1;
        dloss(diter) = dmid.loss;
    end
    
    for k = 1 : gStep
        % 采样
        xBatch = x(:, index((k - 1) * batchSize + 1 : k * batchSize));
        % 生成网络的损失函数计算和梯度更新
        gmid = aaeGeneratorCost(aae, gmid, xBatch);
        aae = aaeGeneratorUpdate(aae, opt, gmid);
        
        giter = giter + 1;
        gloss(giter,1) = gmid.gloss;
        gloss(giter,2) = gmid.dloss;
    end
    
    % 输出
    subplot(1,3,1);
    display_network(ae.w1');
    subplot(1,3,2);
    display_network(ae.w2);
    subplot(1,3,3);
    plot([1:i],mloss(1:i));
end

end


%% 计算动量所需的临时变量
function [gmid, dmid] = aaeInitMidMt(gan)
dSize = numel(gan.dArchitecture);
for n = 1 : (dSize-1)
	
    dmid.dvwDiff{n} = zeros(size(gan.dw{n}));
    dmid.dvbDiff{n} = zeros(size(gan.db{n}));
end
gmid.gvw1 = zeros(size(gan.gw1));
gmid.gvb1 = zeros(size(gan.gb1));
gmid.gvw2 = zeros(size(gan.gw2));
gmid.gvb2 = zeros(size(gan.gb2));
end