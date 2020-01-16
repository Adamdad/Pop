function [v, y] = cameraJointToMarginals(yvInfo, probYV)
% yvInfo is the pYV struct from testCameraObjectGeometryScript, etc.
% probYV is the "marginal" of P(y, v | evidence)
% v.(f,x,ml) gives the possible values of v (x), the likelihoods (f), and
% the most likely value (ml) of the horizon
% y.(f,y,ml) gives the same for the camera height

v.x = yvInfo.v;
v.f = zeros(size(v.x));

y.x = yvInfo.y;
y.f = zeros(size(y.x));

for k = 1:numel(probYV)
    
    vx = yvInfo.vmat(k);
    vind = find(v.x==vx);
    v.f(vind) = v.f(vind) + probYV(k);
    
    yx = yvInfo.ymat(k);
    yind = find(y.x==yx);
    y.f(yind) = y.f(yind) + probYV(k);    
    
end

[tmp, ind] = max(v.f);
v.ml = v.x(ind);
v.exp = sum(v.x .* v.f);

[tmp, ind] = max(y.f);
y.ml = y.x(ind);
y.exp = sum(y.x .* y.f);    