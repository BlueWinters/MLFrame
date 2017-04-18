function nn = nnUpdateGrads(nn)

n = nn.size;
lr = nn.optMethod.rate;
moment = nn.optMethod.moment;

for i = 1:(n-1)
    % 残差梯度
    wDiff = lr * nn.wDiff{i};
    
    % moment动量方法
    if(moment > 0)
        nn.wMoment{i} = moment * nn.wMoment{i} + wDiff;
        wDiff = nn.wMoment{i};
    end
    
    % 更新梯度
    nn.w{i} = nn.w{i} - wDiff;
end

end