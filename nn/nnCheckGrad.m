function nnCheckGrad

nCases = 1000;
nDims = 20;

% 设置网络结构
initnn.architecture = [nDims 15 10 nDims];
initnn.size = size(initnn.architecture, 2);
initnn.layerCfg = cell(initnn.size-1,1);

initnn.layerCfg{2}.actFunc = 'Sigmoid';
initnn.layerCfg{3}.actFunc = 'Sigmoid';
initnn.layerCfg{4}.actFunc = 'Sigmoid';

initnn = nnInitialize(initnn);

% 只用少量样本进行梯度检查
x = rand(20,nCases);
y = x;

epsilon = 1e-4;
diff = cell(initnn.size-1,1);
diffV = cell(initnn.size-1,1);

for i = 1 : (initnn.size-1)
    % 计算参数无变化时候的梯度
    nn = initnn;
    mid = nnFeedforward(nn, x, y);
    mid = nnBackpropagate(nn, mid);
    grad = [mid.wDiff{i}(:) ; mid.bDiff{i}(:)];
    
    % 计算w参数微小变化时候的梯度
    w = initnn.w{i};
    wGrad = zeros(size(w));
    for j = 1 : numel(w)
        % 重新赋值，每次只改变一个参数
        nn = initnn;
        nn.w{i}(j) = w(j) + epsilon;
        mid = nnFeedforward(nn, x, y);
        loss1 = mid.loss;
        
        nn = initnn;
        nn.w{i}(j) = w(j) - epsilon;
        mid = nnFeedforward(nn, x, y);
        loss2 = mid.loss;
        
        wGrad(j) = (loss1 - loss2) / (epsilon*2.0);
    end
    
    
    % 计算b参数微小变化时候的梯度
    b = initnn.b{i};
    bGrad = zeros(size(b));
    for j = 1 : numel(b)
        % 重新赋值，每次只改变一个参数
        nn = initnn;
        nn.b{i}(j) = b(j) + epsilon;
        mid = nnFeedforward(nn, x, y);
        loss1 = mid.loss;
        
        nn = initnn;
        nn.b{i}(j) = b(j) - epsilon;
        mid = nnFeedforward(nn, x, y);
        loss2 = mid.loss;
        
        bGrad(j) = (loss1 - loss2) / (epsilon*2.0);
    end
    
    numgrad = [wGrad(:) ; bGrad(:)];
    diff{i} = numgrad - grad;
    diffV{i} = norm(numgrad - grad) / norm(numgrad + grad);
    disp(['layer diff ' num2str(i) ': ' num2str(diffV{i})]);
end


end