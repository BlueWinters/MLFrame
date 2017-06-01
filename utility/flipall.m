function x = flipall(x)
% 翻转所有的维度
for i=1:ndims(x)
    x = flip(x,i);
end
end