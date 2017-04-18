function mid = cnnFeedforward(cnn, x, y)

% ѵ��������
mid.nCases = size(x,4);

% feature map����
mid.fMaps = cell(cnn.size + cnn.fsize - 1,1);
mid.fMaps{1} = x;

% feature map������
mid.nMaps = zeros(numel(cnn.layers) - 1,1);
mid.nMaps(1) = size(x,3);

% feature map�Ĵ�С
mid.sMaps = zeros(cnn.size - 1,2);
mid.sMaps(1,:) = [size(x,1) size(x,2)];

for n = 2 : (cnn.size-1)
    % ��¼��ǰ�Ĳ���
    mid.n = n;
    
    % �����
    if strcmp(cnn.layers{n}.type, 'conv')
        mid = cnnConvolution(cnn, mid);
    end
    % �����
    if strcmp(cnn.layers{n}.type, 'act')
        mid = cnnActivation(cnn, mid);
    end   
    % �ػ���
    if strcmp(cnn.layers{n}.type, 'pool')
        mid = cnnPooling(cnn, mid);
    end
end

    
% ���һ�������ȫ���Ӳ�
mid.n = cnn.size;
assert(strcmp(cnn.layers{cnn.size}.type, 'fc'), ...
    'error: last layer must be full connect')
mid = cnnFullConnect(cnn, mid);


mid.error = y - mid.fMaps{end};
mid.loss = 1/2 * sum(sum(mid.error.^2)) / mid.nCases;

end