function ae = aeCopy(initae)
% 必须的
ae.v = initae.v;
ae.h = initae.h;
ae.encoder = initae.encoder;
ae.decoder = initae.decoder;

% 可选的
if(isfield(ae,'w1')) ae.w1 = initae.w1; end
if(isfield(ae,'w2')) ae.w2 = initae.w2; end
if(isfield(ae,'b1')) ae.b1 = initae.b1; end
if(isfield(ae,'b2')) ae.b2 = initae.b2; end

end