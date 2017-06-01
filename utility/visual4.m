function visual4(input, order)

assert(numel(size(input)) == 4, 'error input format.');
assert(numel(size(order)) == 2, 'error order.');
assert(prod(order) >= size(input,4), 'error count.');

imrow = size(input,1);
imcol = size(input,2);

chls = size(input,3);
cases = size(input,4);
imsize = [size(input,1) size(input,2)];
bufsize = [imsize.*order + order + 1, chls];

buffer = uint8(zeros(bufsize));
for n = 1 : cases
    i = ceil(n/order(2)) - 1;
    j = mod(n - i*order(2) - 1 + order(2), order(2));
    xidx = 1+imrow*i+(i+1) : 1+imrow*i+(i+1)+imrow-1;
    yidx = 1+imcol*j+(j+1) : 1+imcol*j+(j+1)+imcol-1;
    buffer(xidx, yidx, :) = reshape(input(:,:,:,n), [imsize chls]);
end

imshow(buffer);
drawnow;
end