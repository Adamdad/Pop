detname = 'E:\Matlab2019b\POP\popTestImages\candidate.mat';
detfile = load(detname);
candidate = detfile.candidate ;
candidates=[];
obj_map{1}=1;
obj_map{3}=2;
obj_map{6}=2;
obj_map{8}=2;
mincarh = 32;
minpeoh = 96;
for i=1:size(candidate,2)
    c = candidate{i};
    if isempty(c)
            cand.conf = [];
            cand.bbox =  [];
            cand.p = 0;
            cand.objType =[];
            candi{1}(:, j) = cand;
    else
         for j=1:size(c,2)
            cand.conf =  double(c{j}.conf');
            cand.bbox =  double(uint32(c{j}.bbox));
            y1 = cand.bbox(:,2);
            x2 = cand.bbox(:,3);
            cand.bbox(:,2) =  x2;
            cand.bbox(:,3) = y1;
            cand.p =   double(c{j}.p);
            cand.objType = obj_map{c{j}.ObjType};
            candi{1}(:, j) = cand;
        end
    end

    candidates=[candidates;candi];
    candi = {};
end
