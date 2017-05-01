function mid = cnnFullConnect(cnn, mid)

% 当前层数
n = mid.n;
nCases = mid.nCases;

% 中间变量赋值
% mid.sMaps(n,:) = [];
% mid.nMaps(n) = [];
% h = prod(mid.sMaps{n-1}) * mid.nMaps(n-1);
% v = cnn.layers{n}.layerSet(1);

lastdim = prod(size(mid.fMaps{n-1})) / nCases;

for l = 1 : cnn.fsize
    lidx = n + l - 1;
    mid.z{lidx} = cnn.fcw{l} * reshape(mid.fMaps{lidx-1}, lastdim, nCases) + repmat(cnn.fcb{l},1,nCases);
    mid.fMaps{lidx} = active(mid.z{lidx}, cnn.layers{n}.function);
    lastdim = cnn.layers{n}.layerSet(l);
end


end