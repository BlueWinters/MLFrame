function cae = caeTrain(cae, opt, x, y)

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
cae = aeInitParameters(cae);
% 初始化中间变量
mid = adInitMidMt(cae);

for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = x(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        % 损失函数计算
        mid = caeContractive(cae, mid, xBatch, yBatch);
        % 梯度下降优化参数
        [cae, mid] = aeSgdMomentum(cae, opt, mid);
        
        loss(iter) = mid.loss;
        jacobi(iter) = mid.jacobi;
        iter = iter + 1;
    end

    time = toc;
    
    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    mjacobi(i) = mean(jacobi((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
    display_network(cae.w1');
    subplot(2,2,2);
    display_network(cae.w2);
   
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

%%
function mid = adInitMidMt(ae)
mid.vw1 = zeros(size(ae.w1));
mid.vw2 = zeros(size(ae.w2));
mid.vb1 = zeros(size(ae.b1));
mid.vb2 = zeros(size(ae.b2));
end