function nn = nnSetup(architecture)

% ÅĞ¶ÏÍøÂç½á¹¹Îª[l1 l2 l3 l4 ...]
assert(size(architecture,1) == 1 && size(architecture,2) > 1, ...
    'network architecture error');

nn.architecture = architecture;
nn.size = size(nn.architecture, 2);
nn.layerCfg = cell(nn.size-1,1);

% ÍøÂçÖğ²ãµÄÅäÖÃ
for n = 1 : nn.size
    nn.layerCfg{n}.actFunc = 'Sigmoid';
    nn.layerCfg{n}.dropout = 0;
end

end