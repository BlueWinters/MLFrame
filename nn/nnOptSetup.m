function opt = nnOptSetup()
% �Ż�����
opt.optMethod = @nnSgdMomentum;
opt.momentum = 0.9; 
opt.learnRate = 0.1;
opt.batchSize = 20;
opt.numEpochs = 400;
end
