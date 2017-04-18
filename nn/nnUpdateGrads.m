function nn = nnUpdateGrads(nn, opt, mid)
nn = opt.optMethod(nn, opt, mid);
end