function mid = cnnActivation(cnn, mid)

% ��ǰ����
n = mid.n;

% �м������ֵ
mid.sMaps(n,:) = mid.sMaps(n-1,:);
mid.nMaps(n) = mid.nMaps(n-1);
mid.fMaps{n} = active(mid.fMaps{n-1}, cnn.layers{n}.function);

end