function im2 = makeOverheadView(im, gndmap, horizon, yc, objdet)

[icon{1}, tmp, alpha{1}] = imread('/IUS/vmr7/dhoiem/context/results/CVPR06/cartop.png');
[icon{2}, tmp, alpha{2}] = imread('/IUS/vmr7/dhoiem/context/results/CVPR06/reddot.png');
for t = 1:numel(icon)
    icon{t} = im2double(icon{t}).*repmat(alpha{t}>0, [1 1 3]);
end

f = 1.36*max(size(im));

x1 = -5;  x2 = 5;
z1 = 20;  z2 = 30;

if horizon < 1
    horizon = horizon*size(im, 1); 
end

boxes = cat(1, objdet.bbox);
ind = find(boxes(:, 4))>0;
objdet = objdet(ind);

orient(1) = getOrientations(horizon, yc, f, size(im), objdet, 1);
orient(2) = 0;

offset = size(im, 1);
for k = 1:numel(objdet)
    offset = min(offset, objdet(k).bbox(1, 4));
end
offset = max(offset-5, horizon+1)-horizon;


inpts = [x1/z2*f  yc/z2*f ;  ...
         x2/z2*f  yc/z2*f ;  ...
         x2/z1*f  yc/z1*f ;  ...
         x1/z1*f  yc/z1*f ];

inpts = inpts + repmat([size(im, 2)/2 -offset], [4 1]);

%figure(1), hold off, imshow(im), hold on, plot(inpts([1:end 1], 1)', inpts([1:end 1], 2)'+horizon+offset, 'r')
        
outpts = 100*[0 0 ; 1 0; 1 1; 0 1];

tform = maketform('projective', inpts, outpts);

gndim = im .* repmat(gndmap>0.5, [1 1 3]);
%figure(3), imshow(gndim)
ind = find(gndmap>=0.5);
ind2 = find(gndmap < 0.5);
if 1
for b = 1:3
    meanval = median(gndim(ind + (b-1)*prod(size(gndmap))));
    gndim(ind2 + (b-1)*prod(size(gndmap))) = meanval;
end
end

im2 = imtransform(gndim((horizon + offset):end, :, :), tform, 'bicubic', 'fill', 0);
im2 = imtransform(gndim((horizon + offset):end, :, :), tform, 'bicubic', 'fill', 0, 'size', [size(im2, 1)*2, size(im2, 2)*2]);
im2 =min(max(im2, 0), 1);
size(im2)
hold off, imshow(im2);

minz = yc*f/(size(im, 1)-horizon);
maxz = yc*f/offset;
minx = maxz*(-size(im, 2)/2) / f;
maxx = -minx;
%disp(num2str([minx maxx minz maxz]))

colors = 'gr';
marks = '++';
for k = 1:numel(objdet)
    box = objdet(k).bbox(1, :);
    type = objdet(k).objType;
    v = box([4 4])-horizon;
    u = box(1:2) - size(im, 2)/2;    
    %disp(num2str(u))
    z = yc*f ./ v;
    x = u.*z ./ f;
    %disp(num2str([x z]))
    %disp(num2str([(x-minx)/(maxx-minx)*size(im2, 2) size(im2, 1)-(z-minz)/(maxz-minz)*size(im2, 1)]))
    if (box(2)>0)
        %disp(num2str(k))
        iconw = (x(2)-x(1)) / (maxx-minx) * size(im2, 2);
        tmpim = imrotate(imresize(icon{type}, iconw/size(icon{type}, 2)), orient(type));
        if type==1
            tmpx = iconw/2-cos(orient(type)/180*pi)*iconw/2;
            tmpy = iconw/2+sin(orient(type)/180*pi)*iconw/2;
            %disp(num2str([iconw/2 tmpx orient(type) cos(orient(type)/180*pi)*iconw/2]))
            %figure(3), hold off, imshow(tmpim)
            %figure(3), hold on, plot(tmpx, tmpy, 'r*')
            %pause            
        else
            tmpx = size(tmpim, 2)/2;
            tmpy = size(tmpim, 1)/2;
        end
        %disp(num2str(size(tmpim)))
        %disp(num2str([tmpx tmpy]))

        trans(1) = (size(im2, 1)-(mean(z)-minz)/(maxz-minz)*size(im2, 1)) - tmpy;
        trans(2) = (mean(x)-minx)/(maxx-minx)*size(im2, 2)-tmpx;
        %disp(trans)
        mask = zeros(size(im2));
        mask(1:size(tmpim, 1), 1:size(tmpim, 2), :) = tmpim;
        se = translate(strel(1), round(trans));
        mask2 = max(imdilate(mask,se), 0);
        mask2 = mask2(1:size(im2, 1), 1:size(im2, 2), :);
        %figure(3), imshow(mask2), pause
        ind = find(mask2>0);
        im2(ind) = mask2(ind);
        %im2 = (im2 | mask2);
        %hold on, plot((x-minx)/(maxx-minx)*size(im2, 2), size(im2, 1)-(z-minz)/(maxz-minz)*size(im2, 1), [colors(type) '-' marks(type)], 'LineWidth', 2);        
    end
end
hold off, imshow(im2);


set(gca,'XTick', [1 size(im2, 2)/2 size(im2, 2)])
set(gca,'xTickLabel', round([minx 0 maxx]*10)/10, 'FontSize', 18) 
xlabel('meters')
set(gca,'YTick', [1 size(im2, 1)])
set(gca,'yTickLabel', round([maxz minz]*10)/10, 'FontSize', 18) 
ylabel('meters')
axis on


function orient = getOrientations(h, yc, f, imsize, objdet, t)

orient = 0;
try

    ind = find([objdet.objType]==t);
    boxes = cat(1, objdet(ind).bbox);
    cx = mean(boxes(:, 1:2), 2);
    cy = boxes(:, 4);
    [b, stats] = robustfit(cx, cy);
    ind = find(stats.w>0.9);
    while numel(stats.w)>4
       ind = find(stats.w>0.9);
        [b, stats] = robustfit(cx(ind), cy(ind));
       if numel(stats.w)==ind
           break;
       end
    end    
    %figure(4), hold off, plot(cx(ind), cy(ind), '*')
    %figure(4), hold on, plot([-200 500], b(2)*[-200 500]+b(1), 'r')
    %pause, close(4)
    [tmp, ind2] = max(stats.w);
    cx = cx(ind(ind2))-imsize(2)/2;
    cy = cy(ind(ind2))-h;    
    z1 = yc*f ./ cy;
    x1 = cx.*z1 ./ f; 
    z2 = yc*f ./ (cy+50*b(2));
    x2 = (cx+50).*z2 ./ f;    
    slope = (z2-z1)./(x2-x1);        
    
    %disp(num2str([b(2) slope]))
    
    orient = atan(slope)*180/pi;
catch
end

