function cnn = cnnCheckConfig(cnn, opt, x, y)

% Step1：检查网络的配置
assert(isfield(cnn, 'layers'), 'Config error: cnn.layers miss.');
if(~isfield(cnn, 'size')) cnn.size = numel(cnn.layers); end

% 检查第一层为输入层
cnnCheckConfigInput(cnn, x, y);
% 检查中间的卷积层
cnnCheckConfigMid(cnn, x, y);
% 检查最后层为输出层
cnnCheckConfigOutput(cnn, x, y);
% 检查优化方法
cnnCheckOpt(opt);
end


% 检查输入层
function cnnCheckConfigInput(cnn, x, y)
xsize = size(x);
assert(isfield(cnn.layers{1}, 'type'), 'Config error: layers{%d}.type miss.', 1);
assert(isfield(cnn.layers{1}, 'inputSize'), 'Config error: layers{%d}.inputSize miss.', 1);
assert(isfield(cnn.layers{1}, 'inputChls'), 'Config error: layers{%d}.inputChls miss.', 1);
assert(numel(xsize) == 4, 'Config error: input data error.');

if(~strcmp(cnn.layers{1}.type,'input'))
    error('Config error: layers{%d}.type(%s) error.', 1, cnn.layers{1}.type);
end
if(numel(cnn.layers{1}.inputSize) ~= 2)
    error('Config error: layers{%d}.inputSize(%d) error.', 1, numel(cnn.layers{1}.inputSize));
end
if(cnn.layers{1}.inputSize ~= xsize(1:2))
    error('Config error: layers{%d}.inputSize ~= x(1:2).', 1);
end
if(cnn.layers{1}.inputChls ~= xsize(3))
    error('Config error: layers{%d}.inputChls ~= x(3).', 1);
end
if(size(x,4) ~= size(y,2))
    error('Input error: x and y.');
end
end

% 检查局卷积层
function cnnCheckConfigMid(cnn, x, y)

for n = 2 : (cnn.size-1)
    assert(isfield(cnn.layers{n}, 'type'), 'Config error: layers{%d}.type miss.', n);
    switch cnn.layers{n}.type
        case 'conv'
            assert(isfield(cnn.layers{n}, 'kernelSize'), 'Config error: layers{%d}.kernelSize miss.', n);
            assert(isfield(cnn.layers{n}, 'kernelNums'), 'Config error: layers{%d}.kernelNums miss.', n);
            if(numel(cnn.layers{n}.kernelSize) ~= 2)
                error('Config error: layers{%d}.kernelSize error.', n);
            end
            if(numel(cnn.layers{n}.kernelNums) ~= 1)
                error('Config error: layers{%d}.kernelNums error.', n);
            end
        case 'pool'
            assert(isfield(cnn.layers{n}, 'scaleSize'), 'Config error: layers{%d}.scaleSize miss.', n);
            assert(isfield(cnn.layers{n}, 'scaleType'), 'Config error: layers{%d}.scaleType miss.', n);
            if(numel(cnn.layers{n}.scaleSize) ~= 2)
                error('Config error: layers{%d}.scaleSize error.', n);
            end
            if(~strcmp(cnn.layers{n}.scaleType,'Max') && ~strcmp(cnn.layers{n}.scaleType,'Mean'))
                error('Config error: layers{%d}.scaleType error.', n);
            end
        case 'act'
            assert(isfield(cnn.layers{n}, 'function'), 'Config error: layers{%d}.function miss.', n);
            if(~strcmp(cnn.layers{n}.function, 'ReLU') && ~strcmp(cnn.layers{n}.function, 'Softmax') ...
                    && ~strcmp(cnn.layers{n}.function, 'Softplus') && ~strcmp(cnn.layers{n}.function, 'Linear') ...
                    && ~strcmp(cnn.layers{n}.function, 'Sigmoid'))
                error('Config error: layers{%d}.function(%s) error.', cnn.layers{n}.function);
            end
        otherwise
            error('Config error: unknown layers{%d}.%s.', n, cnn.layers{n}.type);
    end
end
end

%
function cnnCheckConfigOutput(cnn, x, y)
n = cnn.size;
assert(isfield(cnn.layers{n}, 'type'), 'Config error: layers{%d}.type miss.', n);
assert(isfield(cnn.layers{n}, 'layerSet'), 'Config error: layers{%d}.layerSet miss.', n);
assert(isfield(cnn.layers{n}, 'function'), 'Config error: layers{%d}.function miss.', n);

if(~strcmp(cnn.layers{n}.type, 'fc'))
    error('Config error: layers{%d}.type(%s) error.', n, cnn.layers{1}.type);
end
if(numel(cnn.layers{n}.layerSet) == 0)
    error('Config error: layers{%d}.layerSet error.', n);
end
if(~strcmp(cnn.layers{n}.function, 'ReLU') && ~strcmp(cnn.layers{n}.function, 'Softmax') ...
        && ~strcmp(cnn.layers{n}.function, 'Softplus') && ~strcmp(cnn.layers{n}.function, 'Linear') ...
        && ~strcmp(cnn.layers{n}.function, 'Sigmoid'))
    error('Config error: layers{%d}.function(%s) error.', n, cnn.layers{n}.function);
end
if(cnn.layers{n}.layerSet(end) ~= size(y,1))
    error('Config error: layers{%d}.layerSet ~= y error.', n);
end
end

function cnnCheckOpt(opt)
assert(isfield(opt, 'learnRate'), 'Config error: opt.learnRate miss.');
assert(isfield(opt, 'optMethod'), 'Config error: opt.optMethod miss.');
assert(isfield(opt, 'batchSize'), 'Config error: opt.batchSize miss.');
assert(isfield(opt, 'numEpochs'), 'Config error: opt.numEpochs miss.');
end
