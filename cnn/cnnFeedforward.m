function mid = cnnFeedforward(cnn, x, y)

% 训练样本数
mid.nCases = size(x,4);

% feature map数据
mid.fMaps = cell(cnn.size + cnn.fsize - 1,1);
mid.fMaps{1} = x;

% feature map的数量
mid.nMaps = zeros(numel(cnn.layers) - 1,1);
mid.nMaps(1) = size(x,3);

% feature map的大小
mid.sMaps = zeros(cnn.size - 1,2);
mid.sMaps(1,:) = [size(x,1) size(x,2)];

for n = 2 : (cnn.size-1)
    % 记录当前的层数
    mid.n = n;
    
    % 卷积层
    if strcmp(cnn.layers{n}.type, 'conv')
        mid = cnnConvolution(cnn, mid);
    end
    % 激活层
    if strcmp(cnn.layers{n}.type, 'act')
        mid = cnnActivation(cnn, mid);
    end   
    % 池化层
    if strcmp(cnn.layers{n}.type, 'pool')
        mid = cnnPooling(cnn, mid);
    end
end

    
% 最后一层必须是全连接层
mid.n = cnn.size;
assert(strcmp(cnn.layers{cnn.size}.type, 'fc'), ...
    'error: last layer must be full connect')
mid = cnnFullConnect(cnn, mid);


mid.error = y - mid.fMaps{end};
mid.loss = 1/2 * sum(sum(mid.error.^2)) / mid.nCases;

end