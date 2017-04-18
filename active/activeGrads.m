%% 求激活函数的导数1
% y = f(x) --> y' = f'(x)
% 输入
%   x: 自变量x，可以是单变量、矩阵（或多维矩阵）
%   f: 函数类型名称，字符型
function activationGrads = activeGrads(x, func)
    switch func
        case 'Sigmoid'
            activationGrads = sigmoidGradient(x);
        case 'ReLU'
            activationGrads = reluGradient(x);
        case 'Softplus'
            activationGrads = softplusGradient(x);
        case 'Linear'
            activationGrads = linearGradient(x);
    end
end

% ReLU函数的导数(Rectified linear unit)
function activation = reluGradient(x)
    activation = (x>0);
end

% Sigmoid函数的一阶导数
function g = sigmoidGradient(x)
    sig = 1 ./ (1 + exp(-x));
    g = sig.*(1-sig);
end

% Softplus函数的一阶导数
function activation = softplusGradient(x)
    activation = 1 ./ (1+exp(-x));
end

%
function activation = linearGradient(x)
    activation = ones(size(x));
end