function cnn = cnnInitialize(cnn, x)

cnn.size = numel(cnn.layers);
n = cnn.size;

% ����Ĵ�С��x = width * height * channel * cases
mapSize = [size(x,1) size(x,2)]; % width * height
mapNums = size(x,3); % channels
nCases = size(x,4); % cases


% �м����������������map����
inputMaps = mapNums;
outputMaps = 0;

for l = 2 : (n-1)
    % ���
    if strcmp(cnn.layers{l}.type, 'conv')
        ksize = cnn.layers{l}.kernelSize;
        knums = cnn.layers{l}.kernelNums;
        
        % ��һ���map�Ĵ�С
        mapSize = mapSize - ksize + 1;
        assert(all(mapSize > 0), 'Error feature map size.');
        % ��һ���map������
        outputMaps = knums;
        
        % kernel��b��ʼ��
        cnn.kernel{l} = cell(outputMaps, inputMaps);
        cnn.b{l} = cell(outputMaps,1);
        
        fan1 = outputMaps * prod(ksize); % ksize.^2
        for j = 1 : outputMaps  % output map
            fan2 = inputMaps * prod(ksize); % ksize.^2
            for i = 1 : inputMaps  % input map
                cnn.kernel{l}{j,i} = (rand(ksize) - 0.5) * 2 * sqrt(6 / (fan2 + fan1));
            end
            cnn.b{l}{j} = 0;
        end
        
        % ����
%         cnn.nmaps = outputMaps;
        inputMaps = outputMaps;
    end
    
    % �����
    if strcmp(cnn.layers{l}.type, 'act')
        mapSize = mapSize; % ������feature map size��С����
        inputMaps = inputMaps;
        outputMaps = inputMaps;
    end
    
    % �ػ���
    if strcmp(cnn.layers{l}.type, 'pool')
        mapSize = mapSize ./ cnn.layers{l}.scaleSize;
        assert(all(floor(mapSize) == mapSize), 'pool size error.');
        
        inputMaps = inputMaps;
        outputMaps = inputMaps;
    end
end

% ���һ�������ȫ���Ӳ�
cnn.fsize = size(cnn.layers{n}.layerSet, 2);
v = prod(mapSize) * inputMaps;
h = cnn.layers{n}.layerSet(1);
for l = 1 : cnn.fsize
    h = cnn.layers{n}.layerSet(l);
    r  = sqrt(6) / sqrt(h+v+1);
    cnn.fcw{l} = rand(h, v) * 2 * r - r;
    cnn.fcb{l} = zeros(h,1);
    % ��������ȫ���Ӳ�
    v = h;
end


end