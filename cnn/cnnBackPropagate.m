function mid = cnnBackpropagate(cnn, mid)

% feature map�Ĳв�
mid.dfMaps = cell(cnn.size + cnn.fsize - 1,1);
% ����˵��ݶ�
mid.kDiff = cell(size(cnn.kernel));
mid.bDiff = cell(size(cnn.b));
% ȫ����Ȩֵ���ݶ�
mid.fcwDiff = cell(size(cnn.fcw));
mid.fcbDiff = cell(size(cnn.fcb));

% ȫ���ӵ����һ��в���ݶ�
mid.dfMaps{end} = - mid.error .* activeGrads2(mid.fMaps{end}, cnn.layers{end}.function);
mid.fcwDiff{end} = mid.dfMaps{end} * mid.fMaps{end-1}' / mid.nCases;
mid.fcbDiff{end} = mean(mid.dfMaps{end}, 2);

% �в�
for n = (cnn.fsize-1) : -1 : 1
    lfc = cnn.size + n - 1;
    mid.dfMaps{lfc} = (cnn.fcw{n+1}' * mid.dfMaps{lfc+1}) ...
        .* activeGrads2(mid.fMaps{lfc}, cnn.layers{end}.function);
end

% �ݶ�
for n = (cnn.fsize-1) : -1 : 2
    lfc = cnn.size + n - 1;
    mid.fcwDiff{n} = mid.dfMaps{lfc} * mid.fMaps{lfc-1}' / mid.nCases;
    mid.fcbDiff{n} = mean(mid.dfMaps{lfc}, 2); 
end

% ���һ��������ȫ���Ӳ�֮��Ĳв���ݶ�
lastdim = prod(size(mid.fMaps{cnn.size-1})) / mid.nCases;
rsize = [lastdim mid.nCases];
mid.fcwDiff{1} = mid.dfMaps{cnn.size} * reshape(mid.fMaps{cnn.size-1}, rsize)' / mid.nCases;
mid.fcbDiff{1} = mean(mid.dfMaps{cnn.size}, 2);

% ��ȫ���Ӳ����������һ�������
n = cnn.size;
mid.dfMaps{n-1} = reshape((cnn.fcw{1}' * mid.dfMaps{n}), size(mid.fMaps{n-1}));


% �����Ĳв���ݶ�
for n = (cnn.size-1) : -1 : 2
    % ��¼��ǰ�Ĳ���
    mid.n = n;
    
    % �����
    if strcmp(cnn.layers{n}.type, 'conv')
        mid = cnnBpConvolution(cnn, mid);
    end
    % �����
    if strcmp(cnn.layers{n}.type, 'act')
        mid = cnnBpActivation(cnn, mid);
    end
    % �ػ���
    if strcmp(cnn.layers{n}.type, 'pool')
        mid = cnnBpPooling(cnn, mid);
    end
end

end