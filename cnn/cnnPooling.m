function mid = cnnPooling(cnn, mid)

% 当前层数
n = mid.n;

% 中间变量赋值
mid.sMaps(n,:) = mid.sMaps(n-1,:) ./ cnn.layers{n}.scaleSize;
mid.nMaps(n) = mid.nMaps(n-1);
mid.fMaps{n} = zeros(mid.sMaps(n,1), mid.sMaps(n,2), mid.nMaps(n), mid.nCases);

switch cnn.layers{n}.scaleType
    case 'Max'
        mid = cnnPoolingOper(cnn, mid, @matMax);
    case 'Mean'
        mid = cnnPoolingOper(cnn, mid, @matMean);
    otherwise
        error('error pooling type.');
end

end

function mid = cnnPoolingOper(cnn, mid, fucntion)
% 步骤说明：
%   0.pool大小为(4*4)
%   1.重置矩阵大小：input[32 32 3 100]    -->  output[4 8 32 3 100]
%   2.求最大值：    input[4 8 32 3 100]   -->  output[1 8 32 3 100]
%   3.消去维度1：   input[1 8 32 3 100]   -->  output[8 32 3 100]
%   4.重置矩阵大小：input[8 32 3 100]     -->  output[8 4 8 3 100]
%   5.求最大值：    input[8 4 8 3 100]   -->  output[8 1 8 3 100]
%   6.消去维度1：   input[8 1 8 3 100]   -->  output[8 8 3 100]
n = mid.n;
ssize1 = 0; ssize2 = 0;
if(numel(cnn.layers{n}.scaleSize) == 1)
    ssize1 = cnn.layers{n}.scaleSize;
    ssize2 = cnn.layers{n}.scaleSize;
else
    ssize1 = cnn.layers{n}.scaleSize(1);
    ssize2 = cnn.layers{n}.scaleSize(2);
end
ssize = ssize1 * ssize2;

rsize = [ssize1 mid.sMaps(n-1,1)/ssize1 mid.sMaps(n-1,2) mid.nMaps(n) mid.nCases];
z = fucntion(reshape(mid.fMaps{n-1}, rsize), 1);
z = squeeze(z);

rsize = [mid.sMaps(n-1,1)/ssize1 ssize2 mid.sMaps(n-1,2)/ssize2 mid.nMaps(n) mid.nCases];
z = fucntion(reshape(z, rsize), 2);
mid.fMaps{n} = squeeze(z);
end

function xmax = matMax(x, dims)
xmax = max(x, [], dims);
end

function xmean = matMean(x, dims)
xmean = mean(x, dims);
end

