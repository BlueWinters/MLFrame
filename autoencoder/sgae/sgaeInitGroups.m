function group = sgaeInitGroups(ae)

hsize = ae.hidden;
gsize = ae.gsize;

%  ‰»ÎÃıº˛≈–∂œ
assert(mod(hsize, gsize) == 0, ...
    'Error: hidden size %d, group size %d', hsize, gsize);

tmp = repmat([1:gsize], hsize/gsize, 1);
tmp = tmp(:)';
group = full(sparse(1:size(tmp,2), tmp, 1));

end