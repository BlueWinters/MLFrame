function aae = aaeTrain(aae, opt, x, y)

% 临时变量
nCases = size(x,2);
batchSize = nCases/opt.dStep;
numEpochs = opt.numEpochs;
dStep = opt.dStep;
gStep = opt.gStep;
zSize = aae.dArchitecture(1);

dloss = zeros(numEpochs*dStep,1);
mdloss = zeros(numEpochs,1);
gloss = zeros(numEpochs*gStep,2);
mgloss = zeros(numEpochs,2);
diter = 0;
giter = 0;

% 初始化权值
aae = aaeInitParameters(aae);
[gmid, dmid] = aaeInitMidMt(aae);

% 用于随机显示的隐藏层
disCases = 100;
z = rand([aae.dArchitecture(1) disCases]);

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
    
    time = toc;
     
    mdloss(i) = mean(dloss((diter-dStep+1):diter));
    mgloss(i) = mean(gloss((giter-gStep+1):giter));
    
    % 输出
    subplot(2,2,1);
    display_network(aae.gw1');
    subplot(2,2,2);
    display_network(aae.gw2);
    
    subplot(2,2,3);
    plot([1:i],mdloss(1:i));
    axis([0 numEpochs 0 ceil(max(mdloss))]);
    subplot(2,2,4);
    samples = aaeRandSamples(aae,z);
    display_network(samples);
%     plot([1:i],mgloss(1:i,1));
%     axis([0 numEpochs 0 ceil(max(max(mgloss)))]);
end

end


%% 计算动量所需的临时变量
function [gmid, dmid] = aaeInitMidMt(aae)
dSize = numel(aae.dArchitecture);
for n = 1 : (dSize-1)
    dmid.dvwDiff{n} = zeros(size(aae.dw{n}));
    dmid.dvbDiff{n} = zeros(size(aae.db{n}));
end
gmid.gvw1 = zeros(size(aae.gw1));
gmid.gvb1 = zeros(size(aae.gb1));
gmid.gvw2 = zeros(size(aae.gw2));
gmid.gvb2 = zeros(size(aae.gb2));
end

%%
function samples = aaeRandSamples(aae, z)
samples = active(aae.gw2 * z + repmat(aae.gb2, 1, size(z,2)), aae.gDecoder);
end