function cvprGenerateResults(imdir, fn, det1, det2, h1, h2, imnums, outdir, gtruth)
% cvprGenerateResults(imdir, fn, det1, det2, h1, h2, imnums, outdir, gtruth)

% MTF
% cthresh1{1} = [0.26 0.19];  % high thresh, local (car/ppl)
% cthresh1{2} = [0.109 0.103];  % low thresh, local (car/ppl)
% cthresh2{1} = [0.55 0.40];  % high thresh, final (car/ppl)
% cthresh2{2} = [0.276 0.168];  % low thresh, final (car/ppl)

% DALAL
% cthresh1{1} = [0.26 0.19]; % high thresh, local (car/ppl)
% cthresh1{2} = [0.109 0.103]; % low thresh, local (car/ppl)
% cthresh2{1} = [0.81 0.75];   % high thresh, final (car/ppl)
% cthresh2{2} = [0.81 0.75]; % low thresh, final (car/ppl)

% TMP
cthresh1{1} = [0.109 0.103]; % high thresh, local (car/ppl)
cthresh1{2} = [0.109 0.103]; % low thresh, local (car/ppl)
cthresh2{1} = [0.81 0.75];   % high thresh, final (car/ppl)
cthresh2{2} = [0.81 0.75]; % low thresh, final (car/ppl)

%cthresh1{1} = [0.109 0.103];  % high thresh, local (car/ppl)
%cthresh1{2} = [0.109 0.103];  % low thresh, local (car/ppl)
%cthresh2{1} = [0.276 0.168];  % high thresh, final (car/ppl)
%cthresh2{2} = [0.276 0.168];  % low thresh, final (car/ppl)

cnt = 0;

for f = imnums

    cnt = cnt+1;
    disp([num2str(cnt) ': ' fn{f}])
    
    bn = strtok(fn{f}, '.');      
        
    im = imread([imdir '/' fn{f}]);     
    
    figure(1), hold off, imshow(im), hold on
    %plot([1 size(im, 2)], [h1(f) h1(f)]*size(im, 1), 'b-', 'LineWidth', 4);
    %colors = 'gr';    
    colors{1} = [0 1 0 ; 0 1 1];
    colors{2} = [1 0 0 ; 0.95 0.55 0.1];
    for t = 1:2    
        if exist('gtruth')
            gtboxes = gtruth(f, t).bbox;
            used = zeros(1, size(gtboxes, 1))'; 
        end
                       
        %color = colors(t);
        ind = find([det1{t}(:).imgnum]==f);        
        ind2 = find([det1{t}(ind).confidence] > cthresh1{1}(t));        
        ind3 = setdiff(find([det1{t}(ind).confidence] > cthresh1{2}(t)), ind2);          
        for n = ind2
            bbox = det1{t}(ind(n)).bbox;
            if exist('gtruth')
                ov = detComputeOverlap(bbox, gtboxes);
                if any(ov>0.33 & ~used)
                    color = colors{t}(1, :);
                    %[colors(t) '-'];
                    tmp = find(ov>0.33 & ~used);
                    used(tmp(1)) = 1;
                else
                    color = colors{t}(2, :);
                    %color = [colors(t) '--'];
                    %color = ['c'*(t==1)+'o'*(t==2) '-'];
                end
            end
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'linewidth',5, 'Color', color);
            %color(1) = 'k';
            %plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), [color],'linewidth',1);
        end        
        for n = [] %ind3
            bbox = det1{t}(ind(n)).bbox;
            if exist('gtruth')
                ov = detComputeOverlap(bbox, gtboxes);
                if any(ov>0.33 & ~used)
                    color = [colors(t) '-'];
                    tmp = find(ov>0.33 & ~used);
                    used(tmp(1)) = 1;
                else
                    %color = [colors(t) '--'];
                    color = ['c'*(t==1)+'o'*(t==2) '--'];
                end
            end         
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), [color] ,'linewidth',3);
            color(1) = 'k';
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), [color],'linewidth',1);
        end          
        print('-f1', '-depsc2', [outdir '/' bn '_od1_' num2str(t) '.ps'])  
        figure(1), hold off, imshow(im), hold on      
    end    
    %print('-f1', '-depsc2', [outdir '/' bn '_od1.ps'])      
 


    figure(2), hold off, imshow(im), hold on
    plot([1 size(im, 2)], [h2(f) h2(f)]*size(im, 1), 'b-', 'LineWidth', 5);
    %colors = 'gr';    
    for t = 1:2                        
        if exist('gtruth')
            gtboxes = gtruth(f, t).bbox;
            used = zeros(1, size(gtboxes, 1))'; 
        end
                  
        %color = colors(t);
        ind = find([det2{t}(:).imgnum]==f);
        ind2 = find([det2{t}(ind).confidence] > cthresh2{1}(t));        
        ind3 = setdiff(find([det2{t}(ind).confidence] > cthresh2{2}(t)), ind2); 
        for n = ind2
            bbox = det2{t}(ind(n)).bbox;
            if exist('gtruth')
                ov = detComputeOverlap(bbox, gtboxes);
                if any(ov>0.33 & ~used)
                    color = colors{t}(1, :);
                    %color = [colors(t) '-'];
                    tmp = find(ov>0.33 & ~used);
                    used(tmp(1)) = 1;
                else
                    color = colors{t}(2, :);
                    %color = [colors(t) '--'];
                    %color = ['c'*(t==1)+'o'*(t==2) '-'];
                end
            end           
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'linewidth', 5, 'Color', color);
            %color(1) = 'k';
            %plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), [color],'linewidth',1);
        end
        for n = [] %ind3
            bbox = det2{t}(ind(n)).bbox;
            if exist('gtruth')
                ov = detComputeOverlap(bbox, gtboxes);
                if any(ov>0.33 & ~used)
                    color = [colors(t) '--'];
                    tmp = find(ov>0.33 & ~used);
                    used(tmp(1)) = 1;
                else
                    %color = [colors(t) '--'];
                    color = ['c'*(t==1)+'y'*(t==2) '--'];
                end
            end         
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), [color] ,'linewidth',3);
            color(1) = 'k';
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), [color],'linewidth',1);
        end      
        print('-f2', '-depsc2', [outdir '/' bn '_od2_' num2str(t) '.ps'])  
        hold off, imshow(im), hold on
        plot([1 size(im, 2)], [h2(f) h2(f)]*size(im, 1), 'b-', 'LineWidth', 5);
        %pause(1)
    end  
    %print('-f2', '-depsc2', [outdir '/' bn '_od2.ps']) 
    
    %outdir = toutdir;

    %end
    
end
    