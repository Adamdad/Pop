function ov = detComputeOverlap(box, boxes)
% ov = computeOverlap(box, boxes)
% Returns area(box AND boxes) / area(box OR boxes)
% box = [x1 x2 y1 y2]
% boxes(nboxes, 4) = [x1 x2 y1 y2]

nboxes = size(boxes, 1);
ov = zeros(nboxes, 1);

for i = 1:size(boxes, 1)    
    
    boxgt = boxes(i, :);
    bi=[max(box(1),boxgt(1)) ; max(box(3),boxgt(3)) ; min(box(2),boxgt(2)) ; min(box(4),boxgt(4))];

    iw=bi(3)-bi(1)+1;
    ih=bi(4)-bi(2)+1;

    a1 = (boxgt(2)-boxgt(1)+1)*(boxgt(4)-boxgt(3)+1);
    a2 = (box(2)-box(1)+1)*(box(4)-box(3)+1);    
    
    ov(i) = 0;
    if iw>0 & ih>0                
        ua = a1 + a2 - iw*ih;        
        ov(i)=iw*ih/ua;
    end        
    
end