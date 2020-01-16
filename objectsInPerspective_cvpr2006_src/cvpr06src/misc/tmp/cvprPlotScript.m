% cvprPlotScript

DO_LOAD = 0;
CREATE_PLOTS = 0;
PLOT_INIT = 1;
PLOT_FINAL = 1;


if CREATE_PLOTS    
    %bsizes{1} = [0 Inf 0 31.99 ; 0 Inf 32 Inf];
    %bsizes{2} = [0 Inf 0 95.99 ; 0 Inf 96 Inf];
    %bsizes{1} = [0 Inf 28 Inf];
    %bsizes{2} = [0 Inf 92 Inf];    
    
    if PLOT_INIT
        %roc_o = candidates2roc(ts.gtruth(ts.good_inds), candidates, [1 2]);
        roc_o = candidates2roc(ts.gtruth(ts.good_inds), candidates, [1 2]);
    end
    if PLOT_FINAL
        %roc_gov = candidates2roc(ts.gtruth(ts.good_inds), newcandidates, [1 2]);
        roc_gov = candidates2roc(ts.gtruth(ts.good_inds), newcandidates, [1 2]);
    end
end
    
if DO_LOAD
    load '../objdet/data/cvprlabelme/labelme_rocresults_o.mat'
    load '../objdet/data/cvprlabelme/labelme_rocresults_og.mat'
    load '../objdet/data/cvprlabelme/labelme_rocresults_oc.mat'
    load '../objdet/data/cvprlabelme/labelme_rocresults_ocg.mat'
    %load '../objdet/data/labelme_rocresults_ocg_car.mat'
    %load '../objdet/data/labelme_rocresults_ocg_ppl.mat'
end

savedir = '/IUS/vmr7/dhoiem/context/results/CVPR06'

medFS = 20;
bigFS = 20;
smFS = 16;

titles = {'Car Detection', 'Pedestrian Detection'};

% for t = 1:2
%     figure(t), hold off, plot(roc_o(t).fp, roc_o(t).tp, '--r', 'LineWidth', 3)
%     hold on, plot(roc_gov(t).fp, roc_gov(t).tp, '-b', 'LineWidth', 3)
%     axis([0 5 0 1])
%     legend(gca, {'Local', 'Final'}, 'FontSize', medFS, 'Location', 'NorthWest')
%     xlabel('FP per Image', 'FontSize', medFS)
%     ylabel('Detection Rate', 'FontSize', medFS)
%     %title('Dalal-Triggs Local Detector', 'FontSize', bigFS)
%     set(findobj(gca, 'Type', 'Legend'), 'FontSize', medFS)   
%     set(gca, 'FontSize', medFS)
% end

for t = 1:2
    try, close(t), catch, end;
    figure(t), hold off;
%     legends = {};
%    color = colors(t);
%    if PLOT_INIT
% %        plot(roc_o(t).fp, roc_o(t).tp, '--r', 'LineWidth', 3)
%         hold off, plot(roc_o(t,1).fp, roc_o(t,1).tp, '--b', 'LineWidth', 1)
% %        plot(roc_os(t,2).fp, roc_os(t,2).tp, '--g', 'LineWidth', 2)        
% %        legends = {'AllInit', 'SmallInit', 'LargeInit'};
%         legends = {'Init'};
%     end
%     if PLOT_FINAL
% %        plot(roc_gov(t).fp, roc_gov(t).tp, 'r', 'LineWidth', 4)
%         hold on, plot(roc_gov(t,1).fp, roc_gov(t,1).tp, 'b', 'LineWidth', 2)
% %        plot(roc_govs(t,2).fp, roc_govs(t,2).tp, 'g', 'LineWidth', 3)
% %        legends(end+1:end+3) = {'AllFinal', 'SmallFinal', 'LargeFinal'};        
%         legends(end+1) = {'Final'};
%     end
    hold off, plot(detroc_o2(t).fp, detroc_o2(t).tp, '--r', 'LineWidth', 3);
    hold on, plot(detroc_og(t).fp, detroc_og(t).tp, '-c', 'LineWidth', 3);
    hold on, plot(detroc_oc2(t).fp, detroc_oc2(t).tp, '--g', 'LineWidth', 3);
    hold on, plot(detroc_ocg3(t).fp, detroc_ocg3(t).tp, '-b', 'LineWidth', 3);
    set(t,'Color', 'w')
    %set(gca,'YTick',0:0.1:0.9)
    %set(gca,'XTick',0:0.5:5)
    %set(gca,'YTickLabel','0.0| |0.1| |0.2| |0.3| |0.4| |0.5| |0.6| |0.7| |0.8| |0.9| |','FontSize', medFS)
    axis([0 10 0 0.75])
    %legend(gca, legends, 'FontSize', medFS, 'Location', 'NorthWest')
    legend(gca, {'Obj', 'ObjCam', 'ObjCamGeom'}, 'FontSize', medFS, 'Location', 'NorthWest')
    xlabel('FP per Image', 'FontSize', medFS)
    ylabel('Detection Rate', 'FontSize', medFS)
    %title(titles{t}, 'FontSize', bigFS)
%    set(findobj(gca, 'Type', 'Legend'), 'FontSize', medFS)    
    
end
    
    
