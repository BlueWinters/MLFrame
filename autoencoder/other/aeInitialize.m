function ae = aeInitialize(v, h, func, encoder, decoder, tied)

% �����ж�
assert(nargin == 2 || nargin == 6, ...
    'number ofinput arguments must be 2 or 6')

% �����
ae.v = v;
ae.h = h;

% ��ʼ������
ae = aeInitParameters(ae);

% ��ѡ��
if(isexist(ae,'encoder')) ae.encoder = encoder;
else ae.encoder = 'Sigmoid'; end

if(isexist(ae,'decoder')) ae.decoder = decoder; 
else ae.decoder = 'Sigmoid'; end

if(isexist(ae,'tied')) ae.tied = tied; 
else ae.tied = 0; end

if(isexist(ae,'function')) ae.function = func; 
else ae.function = @aeBasic; end

end

