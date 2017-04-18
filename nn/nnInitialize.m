function nn = nnInitialize(nn)

n = nn.size;

for i = 1 : (n-1)
    v = nn.architecture(i);
    h = nn.architecture(i+1);
    r  = sqrt(6) / sqrt(h+v+1);
    nn.w{i} = rand(h, v) * 2 * r - r;
    nn.b{i} = zeros(h,1);
end

end

