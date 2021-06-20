%% This function w.r.t Two Level Weighted Collaborative k-means 
%(TW-Co-k-means) on the 'MultipleFeature' Dataset;
%--------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thanks to Chang-Dong Wang and Dong Huang and Wei-Shi Zheng.    
% The final approach has not finished yet, there are exsiting                                                                
% the other problem, so if you have any question, please contact
% 1538395488@qq.com, thank you. The detail can be seen in the paper
% IJCAI2016-1731 and IEEE TKDE. The Link is ~ .
%---------------------------------------------------------------------------
clear all;
close all;
%--------------------------------------------------------------------------
%load the data
load mfeat-fou.mat;
load mfeat-fac.mat;
load mfeat-kar.mat;
load mfeat-zer.mat;
load mfeat-pix.mat;
load mfeat_mor.mat;
%--------------------------------------------------------------------------
%Set the parameters of the algorithm
cluster_num = 10;   %The number of the clusters
View_num    =  6;     %The number of Views
xishu1          =  0.3;   %nangda
xishu2          =  30;     %beta 
xishu3          =  60;      %alpha
%dv = [76,216,64,47,240]; % the dimensonality of each view.
dv = [76,216,64,47,240,6];
%dv = [76,216,64,240,6];
%dv = [76,216,64];
%dv = [76,240];
%-------------------------------------------------------------------------
%use the cell X to load the dataset.

X{1}   = mfeat_fou;
X{2}   = mfeat_fac;
X{3}   = data_kar;
X{4}   = data_zer;
X{5}   = data_pix;
X{6}   = data_mor;


%----------------------------------------------------------------------------
% Initialization stage, there are stragety of initialization you can
% choose.

% the first data initialization stragety, we are not recommand. 
%for v =1:View_num
%    X{v} = X{v} ./ sum(sum(X{v}));
%end

% the second initialization stragety.
%for v =1:View_num
%    X{v} = normalize(X{v});
%end

% the 3rd initialization stragety.
%for v = 1:View_num
%    X{v} =  X{v}./repmat(sqrt(sum(X{v}.^2,1)),size(X{v},1),1);
%end

% the 4th initialization stragety.
%for v = 1:View_num
%    X{v} = X{v}./repmat(sqrt(sum(X{v}.^2,2)),1,size(X{v},2));
%end

%for v =1:View_num
%    X{v} = zscore(X{v});
%end

for v =1:View_num
    X{v} = mapminmax((X{v})',0,1);
    X{v} = X{v}';
end

%--------------------------------------------------------------------------
X1     = X{1};
X2     = X{2};
X3     = X{3};
X4     = X{4};
X5     = X{5};
X6     = X{6};
X_sum = [X1 X2 X3 X4 X5 ];
maxtime = 10; % the max iteration times
%--------------------------------------------------------------------------
%Start the algorithmn, at the same time, cacluate the time.

for tim =1:maxtime
   tic
fprintf('The Process is begining.\n');
[ index] = TWCoKmeans(X,cluster_num,View_num,xishu1,xishu2,xishu3,dv);
% [index]  = kmeans(X_sum,cluster_num);
   toc
%--------------------------------------------------------------------------  
Data_view_num = size(X{1},1);
% true label
Ground_cluster_elem = [ones(Data_view_num/cluster_num,1);2*ones(Data_view_num/cluster_num,1);3*ones(Data_view_num/cluster_num,1);4*ones(Data_view_num/cluster_num,1);5*ones(Data_view_num/cluster_num,1);6*ones(Data_view_num/cluster_num,1);7*ones(Data_view_num/cluster_num,1);8*ones(Data_view_num/cluster_num,1);9*ones(Data_view_num/cluster_num,1);10*ones(Data_view_num/cluster_num,1)];
%---------------------------------------------------------------------------
%cacluate the  clustering performance evaluation.

[NMI,AMI,AVI,EMI] = ANMI_analytical_11(Ground_cluster_elem,index);
[f_measure,precision,recall]   = compute_f(Ground_cluster_elem,index);
[AR,RI,~]                      = RandIndex(Ground_cluster_elem,index);   
[NMI1]                = nmi(Ground_cluster_elem,index);
fprintf('The NMI of the result of clustering is : %f\n',NMI);
fprintf('The NMI11 of the result of clustering is : %f\n',NMI1);
fprintf('The AMI of the result of clustering is : %f\n',AMI);
fprintf('The AVI of the result of clustering is : %f\n',AVI);
fprintf('The EMI of the result of clustering is : %f\n',EMI);
[cn cr rta rfa mapping] = CalClassificationRate(index,Ground_cluster_elem);
 [ac,~,~]                       =  CalcMetrics(Ground_cluster_elem,index);
fprintf('The classification rate of clustering is : %f\n',cr);
fprintf('The f_measure of clustering is : %f\n',f_measure);
fprintf('The precision of clustering is : %f\n',precision);
fprintf('The recall of clustering is : %f\n',recall);
fprintf('The AR of clustering is : %f\n',AR);
fprintf('The RI of clustering is : %f\n',RI);
fprintf('The Accuracy of clustering is : %f\n',ac);
fprintf('---------------------end-----------------------------\n');
%------save the result----------------------------------------------------
%------save the result---------------------------------------------------
   result(tim,1) = NMI;
   result(tim,12) = NMI1;
   result(tim,2) = AMI;
   result(tim,3) = AVI;
   result(tim,4) = EMI;
   result(tim,5) = f_measure;
   result(tim,6) = precision;
   result(tim,7) = recall;
   result(tim,8) = cr;
   result(tim,9) = AR;
   result(tim,10) = RI;
   result(tim,11) = ac;
end
  
mean_result = mean(result);
fprintf('nmi = %f\n cr = %f\n',mean_result(12),mean_result(8));