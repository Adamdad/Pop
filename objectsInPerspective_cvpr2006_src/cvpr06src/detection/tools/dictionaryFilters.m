function W = dictionaryFilters
% 
% This are the filters we used in:
%
% K. P. Murphy, A. Torralba and W. T. Freeman (2003). 
% Using the forest to see the trees: a graphical model relating features, objects and scenes. 
% Adv. in Neural Information Processing Systems 16 (NIPS), Vancouver, BC, MIT Press. 


[x,y]=meshgrid(-1:1,-1:1);
g = exp(-(x.^2 + y.^2)/.75);

k=0;

k = k+1; W(k).filter = 1; % THIS IS THE LOW PASS FILTER

% Gaussian derivatives
for i=1:6
    t = pi*(i-1)/6;
    gt = (cos(t)*x+sin(t)*y).*g;
    k = k+1; W(k).filter = gt;
end

% Laplacian
k = k+1;  W(k).filter = [-.5 -1 -.5; -1 6 -1; -.5 -1 -.5];

% Corner detector
%k = k+1;  W(k).filter = [.75 -1 .75; -1 -1 -1; .75 -1 .75];
if 1
    k = k+1;  W(k).filter = [-0.0104   -0.0204    0.0771   -0.0206   -0.0106;
        -0.0204   -0.0401    0.1512   -0.0403   -0.0206;
        0.0771    0.1512    0.4315    0.1512    0.0771;
        -0.0206   -0.0403    0.1512   -0.0401   -0.0204;
        -0.0106   -0.0206    0.0771   -0.0204   -0.0104]; W(k).filter = W(k).filter-mean(W(k).filter(:));
    
    % Long edge detectors:
    for i=2:3
        [x,y]=meshgrid(-i:i,-1:1); x= x /i;
        gl = exp(-(x.^2 + y.^2)/20.5);
        gt = (y).*gl;
        
        k = k+1; W(k).filter = gt;
    end
    
    % Long edge detectors:
    for i=2:3
        [x,y]=meshgrid(-i:i,-1:1); x= x /i;
        gl = exp(-(x.^2 + y.^2)/20.5);
        gt = (y).*gl;
        
        k = k+1; W(k).filter = gt';
    end
end

% 
Nfilters = length(W);



for k = 1:Nfilters
  W(k).filter = (W(k).filter)/max(max(abs(W(k).filter)));
end


if 1
    figure
    for k = 1:Nfilters
        W(k).filter = (W(k).filter)/max(max(abs(W(k).filter)));
        subplot(Nfilters, Nfilters, k)
        image(128*(W(k).filter)+128)
        axis('off'); axis('equal')
    end
    colormap(gray(256))
end
