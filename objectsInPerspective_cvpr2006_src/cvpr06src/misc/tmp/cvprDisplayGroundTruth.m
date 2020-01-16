function cvprDisplayGroundTruth(imdir, fn, gtruth, v0, imnums, outdir)
% cvprDisplayGroundTruth(imdir, fn, gtruth, v0, imnums, outdir)



cnt = 0;

for f = imnums

    cnt = cnt+1;
    disp([num2str(cnt) ': ' fn{f}])
    
    bn = strtok(fn{f}, '.');      
        
    im = imread([imdir '/' fn{f}]);     
    
    figure(1), hold off, imshow(im), hold on
    if ~isempty(v0)
        plot([1 size(im, 2)], [v0(f) v0(f)]*size(im, 1), 'b-', 'LineWidth', 4);
    end
    colors{1} = [0 1 0 ; 0 1 1];
    colors{2} = [1 0 0 ; 0.95 0.55 0.1];
    for t = 1:2    
        gtbboxes = gtruth(f, t).bbox;
                                
        for n = 1:size(gtbboxes, 1)
            bbox = gtbboxes(n, :);
            color = colors{t}(1, :);
            plot(bbox([1 1 2 2 1]), bbox([3 4 4 3 3]), 'linewidth',5, 'Color', color);
        end                        
        %figure(1), hold off, imshow(im), hold on      
    end    
    print('-f1', '-depsc2', [outdir '/' bn '_gt.ps']) 
    %imwrite(im, [outdir '/' fn{f}]);
    %print('-f1', '-depsc2', [outdir '/' bn '_od1.ps'])      
end


    