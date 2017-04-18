%% �󼤻���ĵ���2
% y = f(x) --> y' = f'(x)
% ����
%   y: ����ֵy�������ǵ����������󣨻��ά����
%   f: �����������ƣ��ַ���
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

% ReLU�����ĵ���(Rectified linear unit)
function ag = reluGradient2(y)
    ag = (y>0);
end

% Sigmoid������һ�׵���
function ag = sigmoidGradient2(y)
    ag = y.*(1-y);
end

% Softplus������һ�׵���
function ag = softplusGradient2(y)
    ag = 1 - exp(-y);
end

% ���Ժ���
function ag = linearGradient2(y)
    ag = ones(size(y));
end