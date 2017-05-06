function sgae = sgaeTrain(sgae, opt, x, y)

% 临时变量
nCases = size(x, 2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = floor(nCases / batchSize);

loss = zeros(numEpochs*numBatches, 1);
mloss = zeros(numEpochs, 1);
gloss = zeros(numEpochs*numBatches, 1);
mgloss = zeros(numEpochs, 1);

iter = 1;

% 初始化权值
sgae = aeInitParameters(sgae);
% 初始化组信息
group = sgaeInitGroups(sgae);
% 初始化中间变量
mid = aeInitMidMt(sgae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % 损失函数计算
        mid = sgae.function(sgae, group, mid, xBatch, yBatch);
        % 梯度下降优化参数
        sgae = aeSgdMomentum(sgae, opt, mid);
        
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

%%
function mid = aeInitMidMt(ae)
mid.vw1 = zeros(size(ae.w1));
mid.vw2 = zeros(size(ae.w2));
mid.vb1 = zeros(size(ae.b1));
mid.vb2 = zeros(size(ae.b2));
end