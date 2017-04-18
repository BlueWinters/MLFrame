function aeCheckGrad

% ֻ���������������ݶȼ��
nDims = 40;
nHids = 10;
nCases = 1000;
x = rand(nDims, nCases);
y = x;

initae.v = nDims;
initae.h = nHids;
initae.tied = 0;
initae.function = @aeSparse;
initae.sparsity = 0.1;
initae.beta = 3;
initae.weightdecay = 0.01;
initae.encoder = 'Sigmoid';
initae.decoder = 'Linear';

initae = aeInitParameters(initae);

% ��ʱ����
epsilon = 1e-4;
initTheta = aeSerialize(initae);
numgrad = zeros(size(initTheta));

ae = initae;

% û���Ŷ����ݶ�
mid = ae.function(ae, x, y);
grad = [mid.w1Diff(:) ; mid.w2Diff(:) ; mid.b1Diff(:) ; mid.b2Diff(:)];

% ΢С�Ŷ����ݶ�
for i = 1 : numel(initTheta)
    theta = initTheta;
    theta(i) = theta(i) + epsilon;
    ae = aeUnSerialize(ae, theta);
    mid = ae.function(ae, x, y);
    loss1 = mid.loss;
    
    theta = initTheta;
    theta(i) = theta(i) - epsilon;
    ae = aeUnSerialize(ae, theta);
    mid = ae.function(ae, x, y);
    loss2 = mid.loss;
    
    numgrad(i) = (loss1 - loss2) / (epsilon * 2.0);
end

% numgrad = numgrad(1:164);
% grad = grad(1:164);

diff = numgrad - grad;
diffV = norm(numgrad - grad) / norm(numgrad + grad);
disp(diffV);

end