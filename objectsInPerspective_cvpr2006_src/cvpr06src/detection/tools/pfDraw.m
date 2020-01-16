function pfDraw (pf, showLegend)
% 
% drawn in the current axis all the polygons contained in pf
% Also shows the legend.
%
%

hold on

if nargin == 1
    showLegend =1;
end

colorPolygon = 'rgbcmyrgbcmyrgbcmyrgbcmyrgbcmyrgbcmyrgbcmyrgb';

Nobjects = size(pf,2);

for i = 1:Nobjects
    if isfield(pf(i), 'class')
        %if isempty(pf(i).class), continue, end;
        objectTag{i} = pf(i).class;
    elseif isfield(pf(i), 'className')
        objectTag{i} = pf(i).className;
    else
        continue
    end
    
    vertices = pf(i).vertices;
    
    X = vertices(1,:); X = [X X(1)];
    Y = vertices(2,:); Y = [Y Y(1)];
        
    polygonHandle = plot(X, Y, '-s', 'color', colorPolygon(i), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', colorPolygon(i), 'MarkerSize',10, 'lineWidth', 3);
    set(polygonHandle, 'Tag', objectTag{i})
    set(polygonHandle, 'userdata', pf(i))
end

if showLegend
    legend(objectTag, -1)
end

hold off
