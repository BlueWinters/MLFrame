function nnCheckGrad

nCases = 1000;
nDims = 20;

% ��������ṹ
initnn.architecture = [nDims 15 10 nDims];
initnn.size = size(initnn.architecture, 2);
initnn.layerCfg = cell(initnn.size-1,1);

initnn.layerCfg{2}.actFunc = 'Sigmoid';
initnn.layerCfg{3}.actFunc = 'Sigmoid';
initnn.layerCfg{4}.actFunc = 'Sigmoid';

initnn = nnInitialize(initnn);

% ֻ���������������ݶȼ��
x = rand(20,nCases);
y = x;

epsilon = 1e-4;
diff = cell(initnn.size-1,1);
diffV = cell(initnn.size-1,1);

for i = 1 : (initnn.size-1)
    % ��������ޱ仯ʱ����ݶ�
    nn = initnn;
    mid = nnFeedforward(nn, x, y);
    mid = nnBackpropagate(nn, mid);
    grad = [mid.wDiff{i}(:) ; mid.bDiff{i}(:)];
    
    % ����w����΢С�仯ʱ����ݶ�
    w = initnn.w{i};
    wGrad = zeros(size(w));
    for j = 1 : numel(w)
        % ���¸�ֵ��ÿ��ֻ�ı�һ������
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
    
    
    % ����b����΢С�仯ʱ����ݶ�
    b = initnn.b{i};
    bGrad = zeros(size(b));
    for j = 1 : numel(b)
        % ���¸�ֵ��ÿ��ֻ�ı�һ������
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