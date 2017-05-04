function aae = aaeInitParameters(aae)

dSize = size(aae.dArchitecture, 2);

% 初始化生成网络权值
v = aae.gArchitecture(1);
h = aae.gArchitecture(2);
r  = sqrt(6) / sqrt(h+v+1);
aae.gw1 = rand(h, v) * 2 * r - r;
aae.gw2 = rand(v, h) * 2 * r - r;
aae.gb1 = zeros(h,1);
aae.gb2 = zeros(v,1);

% 初始化判别网络权值
for n = 1 : (dSize-1)
    v = aae.dArchitecture(n);
    h = aae.dArchitecture(n+1);
    r  = sqrt(6) / sqrt(h+v+1);
    aae.dw{n} = rand(h, v) * 2 * r - r;
    aae.db{n} = zeros(h, 1);
end

end