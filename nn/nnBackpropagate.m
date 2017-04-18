function mid = nnBackpropagate(nn, mid)

n = size(nn.architecture, 2);

% 输出层残差
cfg = nn.layerCfg{n};
mid.delta{n} = - mid.error .* activeGrads(mid.z{n}, cfg.actFunc);
if(isfield(cfg,'dropout')  && cfg.dropout > 0)
    mid.delta{n} = mid.delta{n} .* mid.dropoutMask{n};
end

% 中间隐层残差
for i = (n-1) : -1 : 2
    cfg = nn.layerCfg{i};
    mid.delta{i} = (nn.w{i}' * mid.delta{i+1}) .* activeGrads(mid.z{i}, cfg.actFunc);
    
    % dropout
    if(isfield(cfg,'dropout') && cfg.dropout > 0)
        mid.delta{i} = mid.delta{i} .* mid.dropoutMask{i};
    end
end

% 使用残差求梯度
for i = 1 : (n-1)
    mid.wDiff{i} = mid.delta{i+1} * mid.act{i}' / size(mid.delta{i+1},2);
    mid.bDiff{i} = mean(mid.delta{i+1},2);
end


end