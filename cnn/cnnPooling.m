function mid = cnnPooling(cnn, mid)

% ��ǰ����
n = mid.n;

% �м������ֵ
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
% ����˵����
%   0.pool��СΪ(4*4)
%   1.���þ����С��input[32 32 3 100]    -->  output[4 8 32 3 100]
%   2.�����ֵ��    input[4 8 32 3 100]   -->  output[1 8 32 3 100]
%   3.��ȥά��1��   input[1 8 32 3 100]   -->  output[8 32 3 100]
%   4.���þ����С��input[8 32 3 100]     -->  output[8 4 8 3 100]
%   5.�����ֵ��    input[8 4 8 3 100]   -->  output[8 1 8 3 100]
%   6.��ȥά��1��   input[8 1 8 3 100]   -->  output[8 8 3 100]
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

