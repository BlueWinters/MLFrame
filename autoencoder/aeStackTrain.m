function sae = aeStackTrain(saeCfg, opt, x, y)

n = size(saeCfg.architecture, 2);
sae = cell(n,1);

% 中间变量
nCases = size(x,2);
mid = x;

for i = 2 : n
    ae = saeCfg.layerCfg{i};
    ae.v = saeCfg.architecture(i-1);
    ae.h = saeCfg.architecture(i);
    
    % 训练一层AE
    ae = sgaeTrain(ae, opt, mid, y);
    
    % 计算下次迭代用的输入
    mid = ae.w1 * mid + repmat(ae.b1,1,nCases);
    mid = active(mid, ae.encoder);
    
    % 返回SAE计算结果
    sae{i} = ae;
end

end
