function ae = aeSgdNormal(ae, opt, mid)

lr = opt.learnRate;

if(ae.tied == 0)
    gradTheta = [mid.w1Diff(:) ; mid.w2Diff(:) ; ...
            mid.b1Diff(:) ; mid.b2Diff(:)];
else
    gradTheta = [mid.w1Diff(:) ; mid.b1Diff(:) ; mid.b2Diff(:)];
end

% ¸üÐÂÌÝ¶È
newTheta = aeSerialize(ae) - lr * gradTheta;
ae = aeUnSerialize(ae, newTheta);

end
