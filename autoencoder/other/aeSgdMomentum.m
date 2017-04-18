function ae = aeSgdMomentum(ae, opt, mid)

lr = opt.learnRate;
mt = opt.momentum;

% ���ڵ�һ�γ�ʼ��
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


% �����ݶ�
mid.vTheta = mt * mid.vTheta + lr * gradTheta;
newTheta = aeSerialize(ae) - mid.vTheta;
ae = aeUnSerialize(ae, newTheta);


end
