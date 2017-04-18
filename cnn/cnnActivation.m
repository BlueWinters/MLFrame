function mid = cnnActivation(cnn, mid)

% 当前层数
n = mid.n;

% 中间变量赋值
mid.sMaps(n,:) = mid.sMaps(n-1,:);
mid.nMaps(n) = mid.nMaps(n-1);
mid.fMaps{n} = active(mid.fMaps{n-1}, cnn.layers{n}.function);

end