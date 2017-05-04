function gan = ganTrain(gan, opt, x, y)

% ��ʱ����
batchSize = opt.batchSize;
numEpochs = opt.numEpochs;
kStep = opt.kStep;

% ��ʼ��Ȩֵ
gan = ganInitParameters(gan);

% ѵ��
for i = 1 : numEpochs
    tic;
    
    for k = 1 : kStep
        % ����
        zBatch = 0;
        xBatch = 0;
        yBatch = 0;
        
        % �б��������ʧ����������ݶȸ���
        dmid = ganDiscriminatorCost(gan, xBatch, yBatch, zBatch);
        gan = ganDiscriminatorUpdate(gan, opt, dmid);
    end
    
    % ����
    z = 0;
    % �����������ʧ����������ݶȸ���
    gmid = ganGeneratorCost(gan, z);
    gan = ganGeneratorUpdate(gan, opt, gmid);
end


end