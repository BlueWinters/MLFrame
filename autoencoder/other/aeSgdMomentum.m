function [ae, mid] = aeSgdMomentum(ae, opt, mid)

if(ae.tied == 0)
    [ae, mid] = aeSgdMomentumNonTied(ae, opt, mid);
else
    [ae, mid] = aeSgdMomentumTied(ae, opt, mid);
end

end

%%
function [ae, mid] = aeSgdMomentumNonTied(ae, opt, mid)
lr = opt.learnRate;
mt = opt.momentum;
% 计算动量
mid.vw1 = mt * mid.vw1 + lr * mid.w1Diff;
mid.vb1 = mt * mid.vb1 + lr * mid.b1Diff;
mid.vw2 = mt * mid.vw2 + lr * mid.w2Diff;
mid.vb2 = mt * mid.vb2 + lr * mid.b2Diff;
% 跟新梯度
ae.w1 = ae.w1 - mid.vw1;
ae.b1 = ae.b1 - mid.vb1;
ae.w2 = ae.w2 - mid.vw2;
ae.b2 = ae.b2 - mid.vb2;
% ae.w1 = ae.w1 - lr * mid.w1Diff;
% ae.b1 = ae.b1 - lr * mid.b1Diff;
% ae.w2 = ae.w2 - lr * mid.w2Diff;
% ae.b2 = ae.b2 - lr * mid.b2Diff;
end

%%
function [ae, mid] = aeSgdMomentumTied(ae, opt, mid)
lr = opt.learnRate;
mt = opt.momentum;
% 计算动量
mid.vw1 = mt * mid.vw1 + lr * (mid.w1Diff + mid.w2Diff');
mid.vb1 = mt * mid.vb1 + lr * mid.b1Diff;
mid.vb2 = mt * mid.vb2 + lr * mid.b2Diff;
% 跟新梯度
ae.w1 = ae.w1 - mid.vw1;
ae.w2 = ae.w1';
ae.b1 = ae.b1 - mid.vb1;
ae.b2 = ae.b2 - mid.vb2;
% ae.w1 = ae.w1 - lr * (mid.w1Diff + mid.w2Diff');
% ae.w2 = ae.w1';
% ae.b1 = ae.b1 - lr * mid.b1Diff;
% ae.b2 = ae.b2 - lr * mid.b2Diff;
end