function [th, a , b, error] = fitRegressionStump(x, z, w, isb);
% [th, a , b] = fitRegressionStump(x, z);
% The regression has the form:
% z = a * (x>th) + b;
%
% where (a,b,th) are so that it minimizes the weighted error:
% error = sum(w * |z - (a*(x>th) + b)|^2) 
%
% x,z and w are vectors of the same length
% x, and z are real values.
% w is a weight of positive values. There is no asumption that it sums to
% one.
% isb - whether features are binary
% atb, 2003
% Modified by Derek Hoiem 2005 to allow binary features


[Nfeatures, Nsamples] = size(x); % Nsamples = Number of thresholds that we will consider
%Nsamples = length(x); % Nsamples = Number of thresholds that we will consider

if ~isb

    [x, j] = sort(x); % this now becomes the thresholds. I assume that all the values are different. If the values are repeated you might need to add some noise.

    th = x;

    z = z(j); w = w(j);

    Szw = cumsum(z.*w); Ezw = Szw(end);
    Sw  = cumsum(w);

    % This is 'a' and 'b' for all posible thresholds:
    b = Szw ./ Sw;
    Sw(Nsamples) = 0; 
    a = (Ezw - Szw) ./ (1-Sw) - b; 
    Sw(Nsamples) = 1;

    % Now, let's look at the error so that we pick the minimum:
    % the error at each threshold is:
    % for i=1:Nsamples
    %     error(i) = sum(w.*(z - ( a(i)*(x>th(i)) + b(i)) ).^2);
    % end
    % but with vectorized code it is much faster but also more obscure code:
    Error = sum(w.*z.^2) - 2*a.*(Ezw-Szw) - 2*b*Ezw + (a.^2 +2*a.*b) .* (1-Sw) + b.^2;

    % Output parameters. Search for best threshold (th):
    [error, k] = min(Error);

    if k == Nsamples
        th = th(k);
    else
        th = (th(k) + th(k+1))/2;
    end
    a = a(k);
    b = b(k);

else
    
    th = 0.5;
    x0ind = find(x==0);
    x1ind = find(x==1);
    b = sum(z(x0ind).*w(x0ind)) / sum(w(x0ind));
    a = (sum(z.*w) - sum(z(x0ind).*w(x0ind))) / (1-sum(w(x0ind))) - b;    
    error = sum(w.*(z - ( a*(x>th) + b) ).^2);
    
end
    