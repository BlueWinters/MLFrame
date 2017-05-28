function [cnn]  = cnnTrain(cnn, opt, data_train, data_valid)

% 临时变量
nCases = size(data_train.x, 4);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
numBatches = nCases / batchSize;

iter = 1;
mloss = zeros(numEpochs, 1);
loss = zeros(numEpochs*numBatches, 1);
accTrain = 0;
accValid = 0;

% 初始化网络权值
cnn = cnnInitialize(cnn, data_train.x);

for i = 1 : numEpochs
    tic;
    
    % 训练
    index = randperm(nCases);
    for idx = 1:numBatches
        xBatch = data_train.x(:, :, :, index((idx - 1) * batchSize + 1 : idx * batchSize));
        yBatch = data_train.y(:, index((idx - 1) * batchSize + 1 : idx * batchSize));
        
        mid = cnnForwardPropagate(cnn, xBatch, yBatch);
        mid = cnnBackPropagate(cnn, mid);
        cnn = cnnUpdateGrads(cnn, opt, mid);
        
        loss(iter) = mid.loss;
        iter = iter + 1;
    end
    
    % 交叉验证
    if exist('data_valid', 'var')
        accTrain = cnnPredict(cnn, data_train);
        accValid = cnnPredict(cnn, data_valid);
    end

    time = toc;

    mloss(i) = mean(loss((iter-numBatches):(iter-1)));
    
    subplot(2,2,1);
    plot([1:i], mloss(1:i), 'b');
    axis([0 numEpochs 0 ceil(max(mloss))]);

    subplot(2,2,2);
    cnnVisualfMaps(cnn, mid, 2)
    
    subplot(2,2,3);
    cnnVisualKernel(cnn, mid, 2);
%     subplot(2,2,4);
%     cnnVisualKernel(cnn, mid, 4);

	disp(['epoch ' num2str(i) '/' num2str(numEpochs) '. '...
        'time ' num2str(time) ' seconds. ' ...
        'mean squared error ' num2str(mloss(i)) '.' ...
        'train/valid ' num2str(accTrain) '/' num2str(accValid)]);
end

end

%%
function cnnVisualKernel(cnn, mid, n)
ksize = prod(cnn.layers{n}.kernelSize);
imapsnum = mid.nMaps(n);
omapsnum = mid.nMaps(n-1);
image = zeros([ksize omapsnum*omapsnum]);

for i = 1 : imapsnum
    for j = 1 : omapsnum
        idx = (i - 1) * omapsnum + j;
        image(:,idx) = cnn.kernel{n}{i,j}(:);
    end
end
display_network(image);
end

%%
function cnnVisualColorKernel(cnn, mid, n)
n = 2;%cnn.size - 1;
ksize = cnn.layers{n}.kernelSize;
imapsnum = mid.nMaps(n-1);%3
omapsnum = mid.nMaps(n);
image = zeros([ksize 3 omapsnum]); 

for i = 1 : omapsnum
    for j = 1 : imapsnum
        image(:,:,j,i) = reshape(cnn.kernel{n}{i,j}(:),ksize);
    end
end
visual(image*255,[6 6]);
end

%%
function cnnVisualfMaps(cnn, mid, n)
idx = 10;
fmaps = squeeze(mid.fMaps{n}(:,:,:,idx));
image = reshape(fmaps, [mid.sMaps(n,1)*mid.sMaps(n,2) mid.nMaps(n)]);
display_network(image);
end