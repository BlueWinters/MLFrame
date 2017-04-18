function nn = nnSgdMomentum(nn, opt, mid)

n = size(nn.architecture, 2);
lr = opt.learnRate;
momentum = opt.momentum;

% ���û�г�ʼ���ͳ�ʼ��momentum����
if(~isfield(mid, 'mt'))
    mid.mt = 1;
    for i = 1:(n-1)
        mid.vwDiff{i} = zeros(size(nn.w{i}));
        mid.vbDiff{i} = zeros(size(nn.b{i}));
    end
end

% �ݶȸ���
for i = 1:(n-1)
    % ���㶯��
    mid.vwDiff{i} = momentum * mid.vwDiff{i} + lr * mid.wDiff{i};
    mid.vbDiff{i} = momentum * mid.vbDiff{i} + lr * mid.bDiff{i};
    
    % �����ݶ�
    nn.w{i} = nn.w{i} - mid.vwDiff{i};
    nn.b{i} = nn.b{i} - mid.vbDiff{i};
end


end