function ae = aeInitParameters(ae)

v = ae.v;
h = ae.h;

r  = sqrt(6) / sqrt(h+v+1);
ae.w1 = rand(h, v) * 2 * r - r;
ae.w2 = rand(v, h) * 2 * r - r;
ae.b1 = zeros(h,1);
ae.b2 = zeros(v,1);

if(isfield(ae,'tied') && ae.tied == 1)
    ae.w2 = ae.w1';
end

end