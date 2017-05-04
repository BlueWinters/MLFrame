function gan = ganCheckConfig(gan, x, y)
isize = size(x,1);
nclass = size(y,1);
assert(gan.genor(1,end) == isize, 'Generator output layer config error.');
assert(gan.disor(1,1) == isize,  'Discriminator input layer config error.');
assert(gan.disor(1,end) == nclass,  'Discriminator output layer config error.');
end