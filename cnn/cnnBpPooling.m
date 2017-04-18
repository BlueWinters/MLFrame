function mid = cnnBpPooling(cnn, mid)

n = mid.n;
switch cnn.layers{n}.scaleType
    case 'Max'
        mid = cnnMaxUnPooling(cnn, mid);
    case 'Mean'
        mid = cnnMeanUnPooling(cnn, mid);
    otherwise
        error('error pooling type.');
end

end

function mid = cnnMeanUnPooling(cnn, mid)

n = mid.n;
chls = size(mid.fMaps{n}, 3);
cases = size(mid.fMaps{n}, 4);
orsize = size(mid.fMaps{n},1);
ocsize = size(mid.fMaps{n},2);
dim = orsize * ocsize;
rsize = cnn.layers{n}.scaleSize;
csize = cnn.layers{n}.scaleSize;

z1 = repmat(reshape(mid.dfMaps{n}, [1 dim chls cases]), rsize, 1);
z2 = repmat(reshape(z1, [orsize*rsize ocsize chls cases]), csize, 1);
mid.dfMaps{n-1} = reshape(z2, [orsize*rsize ocsize*csize chls cases]);

end

function mid = cnnMaxUnPooling(cnn, mid)
end