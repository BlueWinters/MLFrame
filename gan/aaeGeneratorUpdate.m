function gan = aaeGeneratorUpdate(gan, opt, gmid)

lr = opt.learnRate;
mt = opt.momentum;

% 计算动量
gmid.gvw1 = mt * gmid.gvw1 + lr * gmid.w1Diff;
gmid.gvw2 = mt * gmid.gvw2 + lr * gmid.w2Diff;
gmid.gvb1 = mt * gmid.gvb1 + lr * gmid.b1Diff;
gmid.gvb2 = mt * gmid.gvb2 + lr * gmid.b2Diff;

% 更新梯度
gan.gw1 = gan.gw1 - gmid.gvw1;
gan.gw2 = gan.gw2 - gmid.gvw2;
gan.gb1 = gan.gb1 - gmid.gvb1;
gan.gb2 = gan.gb2 - gmid.gvb2;
end