function mid = cnnBackpropagate(cnn, mid)

% feature map的残差
mid.dfMaps = cell(cnn.size + cnn.fsize - 1,1);
% 卷积核的梯度
mid.kDiff = cell(size(cnn.kernel));
mid.bDiff = cell(size(cnn.b));
% 全连接权值的梯度
mid.fcwDiff = cell(size(cnn.fcw));
mid.fcbDiff = cell(size(cnn.fcb));

% 全连接的最后一层残差和梯度
mid.dfMaps{end} = - mid.error .* activeGrads2(mid.fMaps{end}, cnn.layers{end}.function);
mid.fcwDiff{end} = mid.dfMaps{end} * mid.fMaps{end-1}' / mid.nCases;
mid.fcbDiff{end} = mean(mid.dfMaps{end}, 2);

% 残差
for n = (cnn.fsize-1) : -1 : 1
    lfc = cnn.size + n - 1;
    mid.dfMaps{lfc} = (cnn.fcw{n+1}' * mid.dfMaps{lfc+1}) ...
        .* activeGrads2(mid.fMaps{lfc}, cnn.layers{end}.function);
end

% 梯度
for n = (cnn.fsize-1) : -1 : 2
    lfc = cnn.size + n - 1;
    mid.fcwDiff{n} = mid.dfMaps{lfc} * mid.fMaps{lfc-1}' / mid.nCases;
    mid.fcbDiff{n} = mean(mid.dfMaps{lfc}, 2); 
end

% 最后一个卷基层和全连接层之间的残差和梯度
lastdim = prod(size(mid.fMaps{cnn.size-1})) / mid.nCases;
rsize = [lastdim mid.nCases];
mid.fcwDiff{1} = mid.dfMaps{cnn.size} * reshape(mid.fMaps{cnn.size-1}, rsize)' / mid.nCases;
mid.fcbDiff{1} = mean(mid.dfMaps{cnn.size}, 2);

% 与全连接层相连的最后一个卷基层
n = cnn.size;
mid.dfMaps{n-1} = reshape((cnn.fcw{1}' * mid.dfMaps{n}), size(mid.fMaps{n-1}));


% 卷基层的残差和梯度
for n = (cnn.size-1) : -1 : 2
    % 记录当前的层数
    mid.n = n;
    
    % 卷积层
    if strcmp(cnn.layers{n}.type, 'conv')
        mid = cnnBpConvolution(cnn, mid);
    end
    % 激活层
    if strcmp(cnn.layers{n}.type, 'act')
        mid = cnnBpActivation(cnn, mid);
    end
    % 池化层
    if strcmp(cnn.layers{n}.type, 'pool')
        mid = cnnBpPooling(cnn, mid);
    end
end

end