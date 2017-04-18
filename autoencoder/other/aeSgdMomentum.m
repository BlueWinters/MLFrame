function ae = aeSgdMomentum(ae, opt, mid)

lr = opt.learnRate;
mt = opt.momentum;

% 用于第一次初始化
if(~isfield(mid,'mt'))
    mid.mt = 1;
    mid.vTheta = zeros(size(aeSerialize(ae)));
end

if(ae.tied == 0)
    gradTheta = [mid.w1Diff(:) ; mid.w2Diff(:) ; ...
            mid.b1Diff(:) ; mid.b2Diff(:)];
else
    gradTheta = [mid.w1Diff(:) ; mid.b1Diff(:) ; mid.b2Diff(:)];
end


% 更新梯度
mid.vTheta = mt * mid.vTheta + lr * gradTheta;
newTheta = aeSerialize(ae) - mid.vTheta;
ae = aeUnSerialize(ae, newTheta);


end
