function mid = caeContractive(ae, mid, x, y)

nCases = size(x, 2);
mid.a1 = x;

% 前向传播
mid.z2 = ae.w1 * mid.a1 + repmat(ae.b1,1,nCases);
mid.a2 = active(mid.z2, ae.encoder);
mid.z3 = ae.w2 * mid.a2 + repmat(ae.b2,1,nCases);
mid.a3 = active(mid.z3, ae.decoder);

% contractive正则项
derivative = activeGrads(mid.z2, ae.encoder);
forward = bsxfun(@times, sum(ae.w1.^2, 2), derivative.^2);
mid.jacobi = 1/2 * sum(sum(forward)) / nCases;

% 重构误差
mid.error = mid.a1 - mid.a3;
mid.loss = 1/2 * sum(sum(mid.error.^2)) / nCases ...
    + 1/2 * ae.weightdecay * (sum(sum(ae.w1.^2)) + sum(sum(ae.w2.^2))) ...
    + ae.lambda * mid.jacobi;

% 正则项的梯度
z2dev2 = activeGrads(mid.z2, ae.encoder) .* (1 - 2 * active(mid.z2,ae.encoder));
term = bsxfun(@times,  z2dev2.* derivative, sum(ae.w1.^2, 2));
grad = term * mid.a1' + bsxfun(@times, ae.w1, sum(derivative.^2, 2));

% 残差和梯度
mid.delta3 = - mid.error .* activeGrads(mid.z3, ae.decoder);
mid.delta2 = (ae.w2' * mid.delta3) .* activeGrads(mid.z2, ae.encoder);

mid.w1Diff = mid.delta2 * mid.a1' / nCases ...
    + ae.lambda * grad / nCases ...
    + ae.weightdecay * ae.w1;
mid.w2Diff = mid.delta3 * mid.a2' / nCases ...
    + ae.weightdecay * ae.w2;
mid.b1Diff = sum(mid.delta2, 2) / nCases ...
    + ae.lambda * sum(term,2) / nCases;
mid.b2Diff = sum(mid.delta3, 2) / nCases;

end