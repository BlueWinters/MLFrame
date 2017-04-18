function theta = aeSerialize(ae)
if(ae.tied == 0)
    theta = [ae.w1(:) ; ae.w2(:) ; ae.b1(:) ; ae.b2(:)];
else
    theta = [ae.w1(:) ; ae.b1(:) ; ae.b2(:)];
end
end