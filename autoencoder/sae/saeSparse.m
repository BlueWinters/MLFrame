function mid = saeSparse(ae, x, y)

nCases = size(x, 2);
mid.a1 = x;

sparsity = ae.sparsity;
beta = ae.beta;
weightdecay = ae.weightdecay;

% 前向传播
mid.z2 = ae.w1 * mid.a1 + repmat(ae.b1,1,nCases);
mid.a2 = active(mid.z2, ae.encoder);
mid.z3 = ae.w2 * mid.a2 + repmat(ae.b2,1,nCases);
mid.a3 = active(mid.z3, ae.decoder);

% Sparse正则项
rho = sum(mid.a2,2) / nCases;
sparse = sum(sparsity .* log(sparsity./rho) + (1-sparsity) .* log((1-sparsity) ./ (1-rho)));
sterm = beta * (- sparsity ./ rho + (1-sparsity) ./ (1-rho));

% 重构误差
mid.error = mid.a1 - mid.a3;
mid.loss = 1/2 * sum(sum(mid.error.^2)) / nCases ...
    + 1/2 * weightdecay * (sum(sum(ae.w1.^2)) + sum(sum(ae.w2.^2))) ...
    + beta * sparse;

% 残差和梯度
mid.delta3 = - mid.error .* activeGrads(mid.z3, ae.decoder);
mid.delta2 = (ae.w2' * mid.delta3 + repmat(sterm,1,nCases)) .* activeGrads(mid.z2, ae.encoder);

mid.w1Diff = mid.delta2 * mid.a1' / nCases + weightdecay * ae.w1;
mid.w2Diff = mid.delta3 * mid.a2' / nCases + weightdecay * ae.w2;
mid.b1Diff = sum(mid.delta2, 2) / nCases;
mid.b2Diff = sum(mid.delta3, 2) / nCases;

end