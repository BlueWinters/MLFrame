function ae = aeUnSerialize(ae, theta)

v = ae.v;
h = ae.h;

if(ae.tied == 0)
    ae.w1 = reshape(theta(1:h*v), h, v);
    ae.w2 = reshape(theta(h*v+1:2*h*v), v, h);
    ae.b1 = reshape(theta(2*h*v+1:2*h*v+h), h, 1);
    ae.b2 = reshape(theta(2*h*v+h+1:2*h*v+h+v), v, 1);
else
    ae.w1 = reshape(theta(1:h*v), h, v);
    ae.w2 = ae.w1';
    ae.b1 = reshape(theta(h*v+1:h*v+h), h, 1);
    ae.b2 = reshape(theta(h*v+h+1:end), v, 1);
end

end