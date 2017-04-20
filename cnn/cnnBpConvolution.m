function mid = cnnBpConvolution(cnn, mid)

n = mid.n;
mid.dfMaps{n-1} = zeros(size(mid.fMaps{n-1}));

% 计算残差
for i = 1 : mid.nMaps(n-1)
    z = zeros(mid.sMaps(n-1,1), mid.sMaps(n-1,2), 1, mid.nCases);
    for j = 1 : mid.nMaps(n)
        t = convn(mid.dfMaps{n}(:,:,j,:), flipall(cnn.kernel{n}{j,i}), 'full');
        z = z + convn(mid.dfMaps{n}(:,:,j,:), flipall(cnn.kernel{n}{j,i}), 'full');
    end
    mid.dfMaps{n-1}(:,:,i,:) = z;
end

% 计算卷积核的梯度
for j = 1 : mid.nMaps(n)
    df = mid.dfMaps{n}(:,:,j,:);
    mid.bDiff{n}{j} = sum(df(:)) / mid.nCases;
    
    for i = 1 : mid.nMaps(n-1)
        d1 = shiftdim(mid.dfMaps{n}(:,:,j,:),3);
        d2 = shiftdim(mid.fMaps{n-1}(:,:,i,:),3);
        mid.kDiff{n}{j,i} = squeeze(convn(d2, d1, 'valid')) / mid.nCases;
    end
end

end