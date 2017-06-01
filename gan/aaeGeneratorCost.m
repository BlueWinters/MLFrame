function gmid = aaeGeneratorCost(aae, gmid, x)

nCasesX = size(x,2);
dSize = numel(aae.dArchitecture);

% 生成网络部分
z2 = aae.gw1 * x + repmat(aae.gb1, 1, nCasesX);
a2 = active(z2, aae.gEncoder);
z3 = aae.gw2 * a2 + repmat(aae.gb2, 1, nCasesX);
a3 = active(z3, aae.gDecoder);

% 判别网络部分
gmid.a{1} = [a2];
for n = 2 : dSize
    gmid.z{n} = aae.dw{n-1} * gmid.a{n-1} + repmat(aae.db{n-1}, 1, nCasesX);
    gmid.a{n} = active(gmid.z{n}, aae.dActFunc);
end

% 准备生成样本的标签
% 这里将伪样本“视作”真样本进行梯度下降
labels = repmat([1;0], 1, nCasesX);

% 计算生成网络部分的残差
gmid.gerror = x - a3;
gmid.gloss = 1/2 * sum(sum(gmid.gerror.^2)) / nCasesX;
gmid.gRes3 = - gmid.gerror .* activeGrads(z3, aae.gDecoder);
gmid.gRes2 = (aae.gw2' * gmid.gRes3) .* activeGrads(z2, aae.gEncoder);

% 计算判别网络部分的残差
gmid.derror = labels - gmid.a{end};
gmid.dloss = 1/2 * sum(sum(gmid.derror.^2)) / nCasesX;
gmid.dRes{dSize} = - gmid.derror .* activeGrads(gmid.z{dSize}, aae.dActFunc);
for n = (dSize-1) : -1 : 2
    tmp = (aae.dw{n}'*gmid.dRes{n+1});
    tmp2 = active(gmid.z{n}, aae.dActFunc);
    gmid.dRes{n} = (aae.dw{n}'*gmid.dRes{n+1}) .* active(gmid.z{n}, aae.dActFunc);
end
gmid.dRes{1} = aae.dw{1}'*gmid.dRes{2};

% 计算总的残差
delta3 = gmid.gRes3;
delta2 = gmid.gRes2 + 0.0001*gmid.dRes{1} .* activeGrads(z2, aae.gEncoder);

% 计算权值梯度
gmid.w1Diff = delta2 * x' / nCasesX;
gmid.w2Diff = delta3 * a2' / nCasesX;
gmid.b1Diff = sum(delta2, 2) / nCasesX;
gmid.b2Diff = sum(delta3, 2) / nCasesX;

end