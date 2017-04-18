function nn = nnSgdMomentum(nn, opt, mid)

n = size(nn.architecture, 2);
lr = opt.learnRate;
momentum = opt.momentum;

% 如果没有初始化就初始化momentum变量
if(~isfield(mid, 'mt'))
    mid.mt = 1;
    for i = 1:(n-1)
        mid.vwDiff{i} = zeros(size(nn.w{i}));
        mid.vbDiff{i} = zeros(size(nn.b{i}));
    end
end

% 梯度更新
for i = 1:(n-1)
    % 计算动量
    mid.vwDiff{i} = momentum * mid.vwDiff{i} + lr * mid.wDiff{i};
    mid.vbDiff{i} = momentum * mid.vbDiff{i} + lr * mid.bDiff{i};
    
    % 更新梯度
    nn.w{i} = nn.w{i} - mid.vwDiff{i};
    nn.b{i} = nn.b{i} - mid.vbDiff{i};
end


end