function gan = ganInitParameters(gan)

gsize = size(gan.genor, 2);
dsize = size(gan.disor, 2);

% 初始化生成网络权值
for n = 1 : (gsize-1)
    v = gan.genor(n);
    h = gan.genor(n+1);
    r  = sqrt(6) / sqrt(h+v+1);
    gan.gw{n} = rand(h, v) * 2 * r - r;
    gan.gb{n} = zeros(h,1);
end

% 初始化判别网络权值
for n = 1 : (dsize-1)
    v = gan.disor(n);
    h = gan.disor(n+1);
    r  = sqrt(6) / sqrt(h+v+1);
    gan.dw{n} = rand(h, v) * 2 * r - r;
    gan.db{n} = zeros(h,1);
end

end