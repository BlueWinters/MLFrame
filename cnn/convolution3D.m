%% 用于测试卷积方法
cases = 10;
maps1 = rand(5,5,cases);
maps2 = rand(3,3,cases);

% 反向传播卷积
k1 = convn(shiftdim(maps1,2), shiftdim(maps2,2), 'valid');
k1 = squeeze(k1);

%
k2 = zeros(size(k1));
for c = 1 : cases
    d1 = maps1(:,:,c);
    d2 = maps2(:,:,c);
    k2 = k2 + convn(d1, d2, 'valid');
end

k1 == k1