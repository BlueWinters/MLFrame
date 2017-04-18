function mid = cnnConvolution(cnn, mid)

% ��ǰ����
n = mid.n;

% �м������ֵ
mid.sMaps(n,:) = mid.sMaps(n-1,:) - cnn.layers{n}.kernelSize + 1;
mid.nMaps(n) = cnn.layers{n}.kernelNums;
mid.fMaps{n} = zeros(mid.sMaps(n,1), mid.sMaps(n,2), mid.nMaps(n), mid.nCases);

for j = 1 : mid.nMaps(n,:)
    z = zeros(mid.sMaps(n,1), mid.sMaps(n,2), 1, mid.nCases);
    for i = 1 : mid.nMaps(n-1,:)
        f = mid.fMaps{n-1}(:,:,i,:);
        k = cnn.kernel{n}{j,i};
        t = size(mid.fMaps{n});
        z = z + convn(mid.fMaps{n-1}(:,:,i,:), cnn.kernel{n}{j,i}, 'valid');%%
    end
    mid.fMaps{n}(:,:,j,:) = z;
end

end