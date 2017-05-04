function aae = aaeDiscriminatorUpdate(aae, opt, dmid)

% �б���
dSize = numel(aae.dArchitecture);
lr = opt.learnRate;
mt = opt.momentum;

for n = 1 : (dSize-1)
	% ���㶯��
    dmid.dvwDiff{n} = mt * dmid.dvwDiff{n} + lr * dmid.dwDiff{n};
    dmid.dvbDiff{n} = mt * dmid.dvbDiff{n} + lr * dmid.dbDiff{n};
    
    % �����ݶ�
    aae.dw{n} = aae.dw{n} - dmid.dvwDiff{n};
    aae.db{n} = aae.db{n} - dmid.dvbDiff{n};
end

end