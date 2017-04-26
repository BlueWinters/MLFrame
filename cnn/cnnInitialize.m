function cnn = cnnInitialize(cnn, x)

cnn.size = numel(cnn.layers);
n = cnn.size;

% 输入的大小：x = width * height * channel * cases
mapSize = [size(x,1) size(x,2)]; % width * height
mapNums = size(x,3); % channels
nCases = size(x,4); % cases


% 中级变量，输入输出的map数量
inputMaps = mapNums;
outputMaps = 0;

for l = 2 : (n-1)
    % 卷积
    if strcmp(cnn.layers{l}.type, 'conv')
        ksize = cnn.layers{l}.kernelSize;
        knums = cnn.layers{l}.kernelNums;
        
        % 下一层的map的大小
        mapSize = mapSize - ksize + 1;
        assert(all(mapSize > 0), 'Error feature map size.');
        % 下一层的map的数量
        outputMaps = knums;
        
        % kernel和b初始化
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
        
        % 更替
%         cnn.nmaps = outputMaps;
        inputMaps = outputMaps;
    end
    
    % 激活层
    if strcmp(cnn.layers{l}.type, 'act')
        mapSize = mapSize; % 激活层的feature map size大小不变
        inputMaps = inputMaps;
        outputMaps = inputMaps;
    end
    
    % 池化层
    if strcmp(cnn.layers{l}.type, 'pool')
        mapSize = mapSize ./ cnn.layers{l}.scaleSize;
        assert(all(floor(mapSize) == mapSize), 'pool size error.');
        
        inputMaps = inputMaps;
        outputMaps = inputMaps;
    end
end

% 最后一层必须是全连接层
cnn.fsize = size(cnn.layers{n}.layerSet, 2);
v = prod(mapSize) * inputMaps;
h = cnn.layers{n}.layerSet(1);
for l = 1 : cnn.fsize
    h = cnn.layers{n}.layerSet(l);
    r  = sqrt(6) / sqrt(h+v+1);
    cnn.fcw{l} = rand(h, v) * 2 * r - r;
    cnn.fcb{l} = zeros(h,1);
    % 至少两个全连接层
    v = h;
end


end