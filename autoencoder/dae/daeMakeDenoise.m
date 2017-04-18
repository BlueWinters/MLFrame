function nx = daeMakeDenoise(dae, x)

nDims = size(x,1);
nCases = size(x,2);
nx = zeros(size(x));

switch dae.noise 
    case 'gaussian'
        % �����������gaussian����
        nx = x + randn(size(x));
%         for n = 1 : nCases
%             nx(:,n) = x(:,n) + randn(nDims,1);
%         end
        
    case 'binary'
        % ���������ж�
        assert(dae.fraction < 1 || dae.fraction >= 0, ...
            'binary fraction must in [0,1)'); 
        % �����������binary����
         nx = x .* (rand(size(x)) > dae.fraction);
%         for n = 1 : nCases
%             tt = (rand(nDims,1) > dae.fraction);
%             nx(:,n) = x(:,n) .* (rand(nDims,1) > dae.fraction);
%         end
   
    otherwise
        error('Error noise type %s......\n', dae.noise);
end





end