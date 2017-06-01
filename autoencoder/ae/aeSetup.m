function ae = aeSetup(architecture)

% �ж�����ṹΪ[l1 l2 l3 l4 ...]
assert(size(architecture,1) == 1 && size(architecture,2) > 1, ...
    'network architecture error');

ae.visible = architecture(1);
ae.hidden = architecture(2);
ae.size = size(ae.architecture, 2);
ae.layerCfg = cell(ae.size-1,1);

% ������������
for n = 1 : ae.size
    ae.layerCfg{n}.tied = 0;
    ae.layerCfg{n}.function = @aeBasic;
    ae.layerCfg{n}.encoder = 'Sigmoid';
    ae.layerCfg{n}.decoder = 'Sigmoid';
end

end