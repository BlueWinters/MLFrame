function visual2(input, order)

assert(numel(size(input)) == 2, 'error input format.');
assert(numel(size(order)) == 2, 'error order.');
assert(prod(order) >= size(input,2), 'error: too many images.');

vmax = max(max(input));
vmin = min(min(input));
input = (input - vmin) / (vmax - vmin) * 255;

isize = sqrt(size(input,1));
ncase = size(input,2);
finput = reshape(input, [isize isize 1 ncase]);
visual4(finput, order);

end