%% �����
function activation = active(x, func)
    switch func
        case 'Sigmoid'
            activation = sigmoid(x);
        case 'ReLU'
            activation = relu(x);
        case 'Softplus'
            activation = softplus(x);
        case 'Linear'
            activation = linear(x);
    end
end

% ReLU����(Rectified linear unit)
function activation = relu(x)
    activation = x .* (x>0);
end

% Sigmoid����
function activation = sigmoid(x)
    activation = 1 ./ (1 + exp(-x));
end

% Softplus����
function activation = softplus(x)
    activation = log(1+exp(x));
end

% Linear����
function activation = linear(x)
    activation = x;
end