function [cn cr rta rfa mapping] = CalClassificationRate(predict,groundture)
%Input:
%       predict -- class label from algs;
%       groundture -- the real class label;
%Output:
%       cn: total number of classes;
%       cr: classification rate;
%       rta: rate of ture association;
%       rfa: rate of false association;
%       mapping: map the learned categories to true categories, the ith
%                class in learned categories is mapped to mapping(i);

plen = length(predict);
glen = length(groundture);
if plen~=glen
    error('length of learned categories and ture categories must be the same');
end

preidlist = unique(predict);
cn = length(preidlist);

gndidlist = unique(groundture);
gn = length(gndidlist);

mapping = [];

if cn>=glen % the number of learned categories approaches the
                          % number of training cases
    cr = 1;    
else
    cr = 0;
    for i = 1:cn
        curid = preidlist(i);
        curlist = groundture(predict==curid);
        if gn>1
            numofeachbin = hist(curlist,gndidlist);
            [maxbin pos] = max(numofeachbin);
        else
            maxbin = length(curlist);
            pos = 1;
        end
        cr = cr+maxbin(1);% maxbin will be a arrary if the number of 
                          % training cases in associated categories is the 
                          % same, take the first categories for convenient;
        mapping = [mapping gndidlist(pos)];% map the learned categories to 
                                           % true categories;
    end
    cr = cr/glen;
end

premat = zeros(plen,plen);
gndmat = zeros(glen,glen);
for i =1:cn
    premat(predict==preidlist(i),predict==preidlist(i)) = 1;    
end
for i =1:gn    
    gndmat(groundture==gndidlist(i),groundture==gndidlist(i)) = 1;
end

rta = sum(sum(premat(gndmat==1)))/glen^2;
rfa = sum(sum(premat(gndmat==0)))/plen^2;








