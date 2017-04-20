function cnn = cnnUpdateGrads(cnn, opt, mid)

lr = opt.learnRate;
% mt = opt.moment;

% 卷积层梯度更新
for n = 2 : (cnn.size-1)
    if ~strcmp(cnn.layers{n}.type,'conv') continue; end
    
    for j = 1 : mid.nMaps(n)
        for i = 1 : mid.nMaps(n-1)
            cnn.kernel{n}{j,i} = cnn.kernel{n}{j,i} - lr * mid.kDiff{n}{j,i};
        end
        cnn.b{n}{j} = cnn.b{n}{j} - lr * mid.bDiff{n}{j};
    end
    
end

% 全连接层梯度更新
for n = 1 : cnn.fsize
    cnn.fcw{n} = cnn.fcw{n} - lr * mid.fcwDiff{n};
    cnn.fcb{n} = cnn.fcb{n} - lr * mid.fcbDiff{n};
end  

end