function x = flipall(x)
% ��ת���е�ά��
for i=1:ndims(x)
    x = flip(x,i);
end
end