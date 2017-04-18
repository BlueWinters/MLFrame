%% 求激活函数的导数2
% y = f(x) --> y' = f'(x)
% 输入
%   y: 函数值y，可以是单变量、矩阵（或多维矩阵）
%   f: 函数类型名称，字符型
function ag = activeGrads2(y, func)
    switch func
        case 'Sigmoid'
            ag = sigmoidGradient2(y);
        case 'ReLU'
            ag = reluGradient2(y);
        case 'Softplus'
            ag = softplusGradient2(y);
        case 'Linear'
            ag = linearGradient2(y);
    end
end

% ReLU函数的导数(Rectified linear unit)
function ag = reluGradient2(y)
    ag = (y>0);
end

% Sigmoid函数的一阶导数
function ag = sigmoidGradient2(y)
    ag = y.*(1-y);
end

% Softplus函数的一阶导数
function ag = softplusGradient2(y)
    ag = 1 - exp(-y);
end

% 线性函数
function ag = linearGradient2(y)
    ag = ones(size(y));
end