function mid = cnnBpKsparse(cnn, mid)

n = mid.n;
mid.dfMaps{n-1} = mid.mMaps{n}.*mid.dfMaps{n};
end