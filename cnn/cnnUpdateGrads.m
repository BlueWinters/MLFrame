function nn = nnUpdateGrads(nn)

n = nn.size;
lr = nn.optMethod.rate;
moment = nn.optMethod.moment;

for i = 1:(n-1)
    % �в��ݶ�
    wDiff = lr * nn.wDiff{i};
    
    % moment��������
    if(moment > 0)
        nn.wMoment{i} = moment * nn.wMoment{i} + wDiff;
        wDiff = nn.wMoment{i};
    end
    
    % �����ݶ�
    nn.w{i} = nn.w{i} - wDiff;
end

end