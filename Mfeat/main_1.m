%% Multi-View Weighting COLL algorithmn
%  Thanks to ChangDong Wang and dong-huang and Dong Huang 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
clear all;
close all;

%% Load the data set


dataname ='Mfeat.mat';
disp(['Loading ' dataname '...']);
load (dataname,'data','classid');
%c =length(unique(classid)); %the number of clusters
cluster_num = 10;
%% Set the parameters of the algorithm
View_num    =  6;     %The number of Views
xishu1          =  0.42;   %nangda
xishu2          =  40;     %beta 
xishu3          =  8;      %alpha
dv = [76,216,64,47,240,6];
%data =normalize(data);

start=[1,dv(1)+1,dv(1)+dv(2)+1,dv(1)+dv(2)+dv(3)+1,dv(1)+dv(2)+dv(3)+dv(4)+1,dv(1)+dv(2)+dv(3)+dv(4)+dv(5)+1];
finish =[dv(1),dv(1)+dv(2),dv(1)+dv(2)+dv(3),dv(1)+dv(2)+dv(3)+dv(4),dv(1)+dv(2)+dv(3)+dv(4)+dv(5),649];

for i =1:View_num
data_cell{i} = data(:,start(i):finish(i));
end
Data_view_num = size(data_cell{1},1);
Ground_cluster_elem = [ones(Data_view_num/cluster_num,1);2*ones(Data_view_num/cluster_num,1);3*ones(Data_view_num/cluster_num,1);4*ones(Data_view_num/cluster_num,1);5*ones(Data_view_num/cluster_num,1);6*ones(Data_view_num/cluster_num,1);7*ones(Data_view_num/cluster_num,1);8*ones(Data_view_num/cluster_num,1);9*ones(Data_view_num/cluster_num,1);10*ones(Data_view_num/cluster_num,1)];
maxtime = 10;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for tim =1:maxtime
   tic
fprintf('The Process is begining.\n');
[ index] = TWCoKmeans(data_cell,cluster_num,View_num,xishu1,xishu2,xishu3,dv);
   toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[NMI,AMI,AVI,EMI] = ANMI_analytical_11(Ground_cluster_elem,index);
[f_measure,precision,recall]   = compute_f(Ground_cluster_elem,index);
%[AR,RI,~]                      = RandIndex(Ground_cluster_elem,index);               
fprintf('The NMI of the result of clustering is : %f\n',NMI);
fprintf('The AMI of the result of clustering is : %f\n',AMI);
fprintf('The AVI of the result of clustering is : %f\n',AVI);
fprintf('The EMI of the result of clustering is : %f\n',EMI);
[cn cr rta rfa mapping] = CalClassificationRate(index,Ground_cluster_elem);
fprintf('The classification rate of clustering is : %f\n',cr);
%------save the result--------------
%------save the result--------------
   result(tim,1) = NMI;
   result(tim,2) = AMI;
   result(tim,3) = AVI;
   result(tim,4) = EMI;
   result(tim,5) = f_measure;
   result(tim,6) = precision;
   result(tim,7) = recall;
   result(tim,8) = cr;
end
  

