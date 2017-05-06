function mid = daeDenoise(ae, mid, x, nx, y)

nCases = size(x, 2);
mid.a1 = nx;

weightdecay = ae.weightdecay;

% 前向传播
mid.z2 = ae.w1 * mid.a1 + repmat(ae.b1,1,nCases); %带噪声的样本作为网络的输入
mid.a2 = active(mid.z2, ae.encoder);
mid.z3 = ae.w2 * mid.a2 + repmat(ae.b2,1,nCases);
mid.a3 = active(mid.z3, ae.decoder);

% 重构误差
mid.error = x - mid.a3; %带噪声的样本与原样本的重构误差
mid.loss = 1/2 * sum(sum(mid.error.^2)) / nCases ...
    + 1/2 * weightdecay * (sum(sum(ae.w1.^2)) + sum(sum(ae.w2.^2)));

% 残差和梯度
mid.delta3 = - mid.error .* activeGrads(mid.z3, ae.decoder);
mid.delta2 = (ae.w2' * mid.delta3) .* activeGrads(mid.z2, ae.encoder);

mid.w1Diff = mid.delta2 * mid.a1' / nCases + weightdecay * ae.w1;
mid.w2Diff = mid.delta3 * mid.a2' / nCases + weightdecay * ae.w2;
mid.b1Diff = sum(mid.delta2, 2) / nCases;
mid.b2Diff = sum(mid.delta3, 2) / nCases;

end