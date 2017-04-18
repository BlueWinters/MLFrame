function network = nnCopyCfg(nn)

network.architecture = nn.architecture;
network.size = nn.size;
network.layerCfg = nn.layerCfg;
network.w = nn.w;
network.b = nn.b;

% network.rate = 1;
% network.method = @sgd;
% network.batchSize = 100;
% network.numEpochs = 400;
end