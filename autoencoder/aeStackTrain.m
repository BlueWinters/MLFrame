function sae = aeStackTrain(saeCfg, opt, x, y)

n = size(saeCfg.architecture, 2);
sae = cell(n,1);

% �м����
nCases = size(x,2);
mid = x;

for i = 2 : n
    ae = saeCfg.layerCfg{i};
    ae.v = saeCfg.architecture(i-1);
    ae.h = saeCfg.architecture(i);
    
    % ѵ��һ��AE
    ae = sgaeTrain(ae, opt, mid, y);
    
    % �����´ε����õ�����
    mid = ae.w1 * mid + repmat(ae.b1,1,nCases);
    mid = active(mid, ae.encoder);
    
    % ����SAE������
    sae{i} = ae;
end

end
