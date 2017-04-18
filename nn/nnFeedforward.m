function mid = nnFeedforward(nn, x, y)

nCases = size(x, 2);
n = size(nn.architecture, 2);

mid.act{1} = x;
ploss = 0;

for idx = 2 : n
    cfg = nn.layerCfg{idx};
    mid.z{idx} = nn.w{idx-1}*mid.act{idx-1} + repmat(nn.b{idx-1},1,nCases);
    mid.act{idx} = active(mid.z{idx}, cfg.actFunc);
    
    % dropout
    if(isfield(cfg,'dropout') && cfg.dropout > 0)
        assert(cfg.dropout >= 0 && cfg.dropout < 1, 'dropout fraction should be in [0,1).');
        mid.dropoutMask{idx} = (rand(size(mid.act{idx})) > cfg.dropout);
        mid.act{idx} = mid.act{idx} .* mid.dropoutMask{idx};
    end
    
end

mid.error = y - mid.act{n};
mid.loss = 1/2 * sum(sum(mid.error.^2)) / nCases;

end