%% k-sparse network≈‰÷√

nnConfig.architecture = [784 400 16];
nnConfig.size = size(nnConfig.architecture);
nnConfig.layerCfg = cells(nnConfig.size);

idx = 1;
nnConfig.layerCfg{idx}.actFunc = 'sigmoid';
penalty.
nnConfig.layerCfg{idx}.penalty = cell(1);

idx = 2;
nnConfig.layerCfg{idx}.actFunc = 'sigmoid';
nnConfig.layerCfg{idx}.penalty = cell(1);