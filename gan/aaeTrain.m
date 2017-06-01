function aae = aaeTrain(aae, opt, x, y)

% ��ʱ����
nCases = size(x,2);
batchSize = nCases/opt.dStep;
numEpochs = opt.numEpochs;
dStep = opt.dStep;
gStep = opt.gStep;
zSize = aae.dArchitecture(1);

dloss = zeros(numEpochs*dStep,1);
mdloss = zeros(numEpochs,1);
gloss = zeros(numEpochs*gStep,2);
mgloss = zeros(numEpochs,2);
diter = 0;
giter = 0;

% ��ʼ��Ȩֵ
aae = aaeInitParameters(aae);
[gmid, dmid] = aaeInitMidMt(aae);

% ���������ʾ�����ز�
disCases = 100;
z = rand([aae.dArchitecture(1) disCases]);

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
    
    time = toc;
     
    mdloss(i) = mean(dloss((diter-dStep+1):diter));
    mgloss(i) = mean(gloss((giter-gStep+1):giter));
    
    % ���
    subplot(2,2,1);
    display_network(aae.gw1');
    subplot(2,2,2);
    display_network(aae.gw2);
    
    subplot(2,2,3);
    plot([1:i],mdloss(1:i));
    axis([0 numEpochs 0 ceil(max(mdloss))]);
    subplot(2,2,4);
    samples = aaeRandSamples(aae,z);
    display_network(samples);
%     plot([1:i],mgloss(1:i,1));
%     axis([0 numEpochs 0 ceil(max(max(mgloss)))]);
end

end


%% ���㶯���������ʱ����
function [gmid, dmid] = aaeInitMidMt(aae)
dSize = numel(aae.dArchitecture);
for n = 1 : (dSize-1)
    dmid.dvwDiff{n} = zeros(size(aae.dw{n}));
    dmid.dvbDiff{n} = zeros(size(aae.db{n}));
end
gmid.gvw1 = zeros(size(aae.gw1));
gmid.gvb1 = zeros(size(aae.gb1));
gmid.gvw2 = zeros(size(aae.gw2));
gmid.gvb2 = zeros(size(aae.gb2));
end

%%
function samples = aaeRandSamples(aae, z)
samples = active(aae.gw2 * z + repmat(aae.gb2, 1, size(z,2)), aae.gDecoder);
end