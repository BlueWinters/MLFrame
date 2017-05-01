function mid = cnnPooling(cnn, mid)

% 当前层数
n = mid.n;

% 中间变量赋值
mid.sMaps(n,:) = mid.sMaps(n-1,:) ./ cnn.layers{n}.scaleSize;
mid.nMaps(n) = mid.nMaps(n-1);
mid.fMaps{n} = zeros(mid.sMaps(n,1), mid.sMaps(n,2), mid.nMaps(n), mid.nCases);

% 步骤说明：
%   0.pool大小为(4*4)
%   1.重置矩阵大小：input[32 32 3 100]    -->  output[4 8 32 3 100]
%   2.求最大值：    input[4 8 32 3 100]   -->  output[1 8 32 3 100]
%   3.消去维度1：   input[1 8 32 3 100]   -->  output[8 32 3 100]
%   4.重置矩阵大小：input[8 32 3 100]     -->  output[8 4 8 3 100]
%   5.求最大值：    input[8 4 8 3 100]   -->  output[8 1 8 3 100]
%   6.消去维度1：   input[8 1 8 3 100]   -->  output[8 8 3 100]
n = mid.n;
ssize1 = cnn.layers{n}.scaleSize(1);
ssize2 = cnn.layers{n}.scaleSize(2);
ssize = ssize1 * ssize2;

switch cnn.layers{n}.scaleType
    case 'Max'
        mid = cnnPoolingMax(cnn, mid, ssize1, ssize2);
    case 'Mean'
        mid = cnnPoolingMean(cnn, mid, ssize1, ssize2);
    otherwise
        error('error pooling type.');
end

end

% 最大值池化
function mid = cnnPoolingMax(cnn, mid, ssize1, ssize2)
n = mid.n;

ssize1 = cnn.layers{n}.scaleSize(1);
ssize2 = cnn.layers{n}.scaleSize(2);
fsize1 = mid.sMaps(n,1);
fsize2 = mid.sMaps(n,2);

mid.maxPos{n} = zeros(size(mid.fMaps{n-1}));

for i = 1 : fsize1
    for j = 1 : fsize2
        ridx = (i-1) * ssize1 + 1 : i * ssize1;
        cidx = (j-1) * ssize2 + 1 : j * ssize2;
        mesh = reshape(mid.fMaps{n-1}(ridx,cidx,:,:), [ssize1*ssize2 mid.nMaps(n) mid.nCases]);
        maxv = max(mesh, [], 1);
        % feature赋值
        mid.fMaps{n}(i,j,:,:) = maxv;
        mask = double(repmat(maxv, ssize1*ssize2, 1) == mesh);
        mask = mask ./ repmat(sum(mask, 1), ssize1*ssize2, 1);
        % 该层的mask
        mid.maxPos{n}(ridx,cidx,:,:) = reshape(mask, [ssize1 ssize2 mid.nMaps(n) mid.nCases]);
    end
end

end

% 均值值池化
function mid = cnnPoolingMean(cnn, mid, ssize1, ssize2)
n = mid.n;
rsize = [ssize1 mid.sMaps(n-1,1)/ssize1 mid.sMaps(n-1,2) mid.nMaps(n) mid.nCases];
z = mean(reshape(mid.fMaps{n-1}, rsize), 1);
z = squeeze(z);

rsize = [mid.sMaps(n-1,1)/ssize1 ssize2 mid.sMaps(n-1,2)/ssize2 mid.nMaps(n) mid.nCases];
z = mean(reshape(z, rsize), 2);
mid.fMaps{n} = squeeze(z);
end
