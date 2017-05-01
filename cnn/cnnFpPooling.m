function mid = cnnPooling(cnn, mid)

% ��ǰ����
n = mid.n;

% �м������ֵ
mid.sMaps(n,:) = mid.sMaps(n-1,:) ./ cnn.layers{n}.scaleSize;
mid.nMaps(n) = mid.nMaps(n-1);
mid.fMaps{n} = zeros(mid.sMaps(n,1), mid.sMaps(n,2), mid.nMaps(n), mid.nCases);

% ����˵����
%   0.pool��СΪ(4*4)
%   1.���þ����С��input[32 32 3 100]    -->  output[4 8 32 3 100]
%   2.�����ֵ��    input[4 8 32 3 100]   -->  output[1 8 32 3 100]
%   3.��ȥά��1��   input[1 8 32 3 100]   -->  output[8 32 3 100]
%   4.���þ����С��input[8 32 3 100]     -->  output[8 4 8 3 100]
%   5.�����ֵ��    input[8 4 8 3 100]   -->  output[8 1 8 3 100]
%   6.��ȥά��1��   input[8 1 8 3 100]   -->  output[8 8 3 100]
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

% ���ֵ�ػ�
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
        % feature��ֵ
        mid.fMaps{n}(i,j,:,:) = maxv;
        mask = double(repmat(maxv, ssize1*ssize2, 1) == mesh);
        mask = mask ./ repmat(sum(mask, 1), ssize1*ssize2, 1);
        % �ò��mask
        mid.maxPos{n}(ridx,cidx,:,:) = reshape(mask, [ssize1 ssize2 mid.nMaps(n) mid.nCases]);
    end
end

end

% ��ֵֵ�ػ�
function mid = cnnPoolingMean(cnn, mid, ssize1, ssize2)
n = mid.n;
rsize = [ssize1 mid.sMaps(n-1,1)/ssize1 mid.sMaps(n-1,2) mid.nMaps(n) mid.nCases];
z = mean(reshape(mid.fMaps{n-1}, rsize), 1);
z = squeeze(z);

rsize = [mid.sMaps(n-1,1)/ssize1 ssize2 mid.sMaps(n-1,2)/ssize2 mid.nMaps(n) mid.nCases];
z = mean(reshape(z, rsize), 2);
mid.fMaps{n} = squeeze(z);
end
