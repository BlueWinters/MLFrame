function mid = cnnBpActivation(cnn, mid)
n = mid.n;
mid.dfMaps{n-1} = mid.dfMaps{n} .* activeGrads2(mid.fMaps{n}, cnn.layers{n}.function);
end