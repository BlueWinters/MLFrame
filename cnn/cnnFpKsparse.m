function mid = cnnFpKsparse(cnn, mid)

n = mid.n;
nCases = size(mid.fMaps{1},4);
k = cnn.layers{n}.kSparse;

% 中间变量
mid.fMaps{n} = mid.fMaps{n-1};
smap = size(mid.fMaps{n});
mid.mMaps{n} = ones(size(mid.fMaps{n}));

% K-Sparse: https://arxiv.org/pdf/1409.2752.pdf
% 只在不同样本间实现K-Sparse
[~, order] = sort(mid.fMaps{n}, 4, 'descend');

for i = 1 : smap(1)
    for j = 1 : smap(2)
        for chls = smap(3): -1 : 2
            mid.mMaps{n}(i,j,chls,order(i,j,chls,:)) = 0;
        end
    end
end

mid.fMaps{n} = mid.mMaps{n}.*mid.fMaps{n};

end