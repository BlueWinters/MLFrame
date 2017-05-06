function mid = sgaeGroupW(ae, group, mid, x, y)

nCases = size(x, 2);
mid.a1 = x;

glambda = ae.glambda;
weightdecay = ae.weightdecay;

% 前向传播
mid.z2 = ae.w1 * mid.a1 + repmat(ae.b1,1,nCases);
mid.a2 = active(mid.z2, ae.encoder);
mid.z3 = ae.w2 * mid.a2 + repmat(ae.b2,1,nCases);
mid.a3 = active(mid.z3, ae.decoder);

% weight sparse正则项
mid.groupw = sqrt(group'*sum(ae.w1.^2,2)) + sqrt(group'*sum(ae.w2.^2,2));
% mid.ghidden = sqrt(group'*(mid.a2.^2));
% mid.gloss = sum(sum(mid.ghidden)) / nCases;

% 重构误差
mid.error = mid.a1 - mid.a3;
mid.loss = 1/2 * sum(sum(mid.error.^2)) / nCases ...
    + 1/2 * weightdecay * (sum(sum(ae.w1.^2)) + sum(sum(ae.w2.^2))) ...
    + glambda * sum(mid.groupw);

% Sparse Group正则项的梯度
% zindex = (mid.ghidden == 0);
% divghidden = 1 ./ mid.ghidden;
% divghidden(zindex) = 0;
% sterm = group * divghidden .* mid.a2 .* activeGrads(mid.z2, ae.encoder); %divghidden=1.mid.ghidden


% 残差和梯度
mid.delta3 = - mid.error .* activeGrads(mid.z3, ae.decoder);
mid.delta2 = (ae.w2' * mid.delta3) .* activeGrads(mid.z2, ae.encoder);

mid.w1Diff = mid.delta2 * mid.a1' / nCases ...
	+ glambda * sterm * mid.a1' ...
    + weightdecay * ae.w1;
mid.w2Diff = mid.delta3 * mid.a2' / nCases ...
    + weightdecay * ae.w2;
mid.b1Diff = sum(mid.delta2, 2) / nCases ...
    + glambda * sum(sterm,2);
mid.b2Diff = sum(mid.delta3, 2) / nCases;

end