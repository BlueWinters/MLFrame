%% �󼤻���ĵ���1
% y = f(x) --> y' = f'(x)
% ����
%   x: �Ա���x�������ǵ����������󣨻��ά����
%   f: �����������ƣ��ַ���
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

% ReLU�����ĵ���(Rectified linear unit)
function activation = reluGradient(x)
    activation = (x>0);
end

% Sigmoid������һ�׵���
function g = sigmoidGradient(x)
    sig = 1 ./ (1 + exp(-x));
    g = sig.*(1-sig);
end

% Softplus������һ�׵���
function activation = softplusGradient(x)
    activation = 1 ./ (1+exp(-x));
end

%
function activation = linearGradient(x)
    activation = ones(size(x));
end