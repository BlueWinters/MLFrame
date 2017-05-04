function aae = aaeDiscriminatorUpdate(aae, opt, dmid)

% 判别器
dSize = numel(aae.dArchitecture);
lr = opt.learnRate;
mt = opt.momentum;

for n = 1 : (dSize-1)
	% 计算动量
    dmid.dvwDiff{n} = mt * dmid.dvwDiff{n} + lr * dmid.dwDiff{n};
    dmid.dvbDiff{n} = mt * dmid.dvbDiff{n} + lr * dmid.dbDiff{n};
    
    % 更新梯度
    aae.dw{n} = aae.dw{n} - dmid.dvwDiff{n};
    aae.db{n} = aae.db{n} - dmid.dvbDiff{n};
end

end