function dmid = aaeDiscriminatorCost(aae, dmid, x, z)

nCasesX = size(x,2);
nCasesZ = size(z,2);
nCases = nCasesX + nCasesZ;

% 生成器生成样本（负样本）
z2 = aae.gw1 * x + repmat(aae.gb1, 1, nCasesZ);
a2 = active(z2, aae.gEncoder);
% z3 = aae.gw2 * a2 + repmat(aae.gb2, 1, nCasesZ);
% a3 = active(z3, aae.gDecoder);

% 判别器
dSize = numel(aae.dArchitecture);
% 准备（正/负）样本
dmid.a{1} = [a2 z];
labels = [repmat([0;1], 1, nCasesZ) repmat([1;0], 1, nCasesX)];

for n = 2 : dSize
    dmid.z{n} = aae.dw{n-1} * dmid.a{n-1} + repmat(aae.db{n-1}, 1, nCases);
    dmid.a{n} = active(dmid.z{n}, aae.dActFunc);
end

% 计算D的损失
dmid.error = labels - dmid.a{end};
dmid.loss = 1/2 * sum(sum(dmid.error.^2)) / nCases;

dmid.delta{dSize} = - dmid.error .* activeGrads(dmid.z{dSize}, aae.dActFunc);
for n = (dSize-1) : -1 : 2
    dmid.delta{n} = (aae.dw{n}'*dmid.delta{n+1}) .* active(dmid.z{n}, aae.dActFunc);
end

% 求D参数的梯度
for n = 1 : (dSize-1)
    dmid.dwDiff{n} = dmid.delta{n+1} * dmid.a{n}' / size(dmid.delta{n+1},2);
    dmid.dbDiff{n} = mean(dmid.delta{n+1},2);
end

end