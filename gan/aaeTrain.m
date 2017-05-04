function aae = aaeTrain(aae, opt, x, y)

% ��ʱ����
nCases = size(x,2);
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
dStep = opt.kStep;
gStep = opt.gStep;
zSize = aae.dArchitecture(1);

dloss = zeros(numEpochs*dStep,1);
gloss = zeros(numEpochs*gStep,2);
diter = 0;
giter = 0;

% ��ʼ��Ȩֵ
aae = aaeInitParameters(aae);
[gmid, dmid] = aaeInitMidMt(aae);

% ѵ��
for i = 1 : numEpochs
    tic;
    
    index = randperm(nCases);
    for k = 1 : dStep
        % ����
        zBatch = rand(zSize, batchSize);
        xBatch = x(:, index((k - 1) * batchSize + 1 : k * batchSize));
        
        % �б��������ʧ����������ݶȸ���
        dmid = aaeDiscriminatorCost(aae, dmid, xBatch, zBatch);
        aae = aaeDiscriminatorUpdate(aae, opt, dmid);
        
        diter = diter + 1;
        dloss(diter) = dmid.loss;
    end
    
    for k = 1 : gStep
        % ����
        xBatch = x(:, index((k - 1) * batchSize + 1 : k * batchSize));
        % �����������ʧ����������ݶȸ���
        gmid = aaeGeneratorCost(aae, gmid, xBatch);
        aae = aaeGeneratorUpdate(aae, opt, gmid);
        
        giter = giter + 1;
        gloss(giter,1) = gmid.gloss;
        gloss(giter,2) = gmid.dloss;
    end
    
    % ���
    subplot(1,3,1);
    display_network(ae.w1');
    subplot(1,3,2);
    display_network(ae.w2);
    subplot(1,3,3);
    plot([1:i],mloss(1:i));
end

end


%% ���㶯���������ʱ����
function [gmid, dmid] = aaeInitMidMt(gan)
dSize = numel(gan.dArchitecture);
for n = 1 : (dSize-1)
	
    dmid.dvwDiff{n} = zeros(size(gan.dw{n}));
    dmid.dvbDiff{n} = zeros(size(gan.db{n}));
end
gmid.gvw1 = zeros(size(gan.gw1));
gmid.gvb1 = zeros(size(gan.gb1));
gmid.gvw2 = zeros(size(gan.gw2));
gmid.gvb2 = zeros(size(gan.gb2));
end