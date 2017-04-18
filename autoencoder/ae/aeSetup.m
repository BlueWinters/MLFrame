function ae = aeSetup(architecture)

% ÅĞ¶ÏÍøÂç½á¹¹Îª[l1 l2 l3 l4 ...]
assert(size(architecture,1) == 1 && size(architecture,2) > 1, ...
    'network architecture error');

ae.architecture = architecture;
ae.size = size(ae.architecture, 2);
ae.layerCfg = cell(ae.size-1,1);

% ÍøÂçÖğ²ãµÄÅäÖÃ
for n = 1 : ae.size
    ae.layerCfg{n}.tied = 0;
    ae.layerCfg{n}.function = @aeBasic;
    ae.layerCfg{n}.encoder = 'Sigmoid';
    ae.layerCfg{n}.decoder = 'Sigmoid';
end

end