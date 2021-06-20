%% This function w.r.t Two Level Weighted Collaborative k-means 
%(TW-Co-k-means) on the 'Image Segementation.' Dataset;
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
% Load the data set
clear all;
close all;

load im_se_1.mat;
load im_se_2.mat;
X{1}   = im_se_1;
X{2}   = im_se_2;
%---------------------------------------------------------------------------
%Set the parameters of the algorithm
c =  7;   %The number of the clusters
View_num    =  2;     %The number of Views

%% Set the parameters of the algorithm
coefficient1          =  0.45;    %nangfda
coefficient2          =  50;   %beta
coefficient3          =  60;    %alpha
dv = [9,10];


%----------------------------------------------------------------------------
% Initialization stage, there are stragety of initialization you can
% choose.

% the first data initialization stragety, we are not recommand. 
%for v =1:View_num
%    X{v} = X{v} ./ sum(sum(X{v}));
%end

% the second initialization stragety.
for v =1:View_num
    X{v} = normalize(X{v});
end

max_time = 100;

for tim =1 :max_time
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('The Process is begining.\n');
tic

[ index] = TWCoKmeans(X,c,View_num,coefficient1,coefficient2,coefficient3,dv);

toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_view_num = size(X{1},1);
Ground_cluster_elem = [ones(Data_view_num/c,1);2*ones(Data_view_num/c,1);3*ones(Data_view_num/c,1);4*ones(Data_view_num/c,1);5*ones(Data_view_num/c,1);6*ones(Data_view_num/c,1);7*ones(Data_view_num/c,1,1)];

%----------------------------------------------------------------------------
%cacluate the  clustering performance evaluation.

[NMI,AMI,AVI,EMI] = ANMI_analytical_11(Ground_cluster_elem,index);
[NMI1]                = nmi(Ground_cluster_elem,index);
[f_measure,precision,recall]   = compute_f(Ground_cluster_elem,index);
[AR,RI,~]                      = RandIndex(Ground_cluster_elem,index);               
fprintf('The NMI of the result of clustering is : %f\n',NMI);
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
   result(tim,12) = NMI1;
end
 mean_result = mean(result); fprintf('nmi = %f\n cr = %f\n',mean_result(12),mean_result(8));