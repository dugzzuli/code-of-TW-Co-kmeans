
function [ index ] = TWCoKmeans( X,Cluster_num,View_num,xishu1,xishu2,xishu3,dv)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thanks to ChangDong Wang and Dong Huang,written by GuangYu Zhang.   
% We proposed a new multi-view clustering which called Multi-View
% Collaborative Kmeans algorithm.It extended the traditional K-means
% method to multi-view using the Collabroative strategy.
% X -- Dataset(sample space).
% Cluster_num -- the number of  the clusters
% View_num -- the number of the View space
% xishu1,xishu2 -- two facors of the function 
%setting the parameters of function respectively controlling the penalty
%terms and the entropy term of View vector.

%--------------------------------------------------------------------------
% Replacing the parameters with some characters. 
T = View_num;
K = Cluster_num;
nangda = xishu1;
alf    = xishu2;
beta   = xishu3;
data_n = size(X{1},1);
%--------------------------------------------------------------------------
% Initializing the variable  'cluster center' and 'V'(view weighting),
%'W'(feature weighting) as in paper .

%--------------------------------------------------------------------------
%initialize the cluster center
initial = randperm(data_n,K);

for t = 1:T
    center{t} = X{t}(initial,:);
end
%--------------------------------------------------------------------------
% initialize V 
for v =1:View_num
    V = ones(1,View_num)./ View_num;
end
%--------------------------------------------------------------------------
% initialize W 
 for v =1:View_num
     
   W{v} =ones(1,dv(v))./dv(v);   
 
 end 
 for t =1:T
    for k = 1:K
       center_sum = repmat(center{t}(k,:),data_n,1);
       D{t}(:,k)  = (X{t} - center_sum).^2*W{t}';              
    end
end

%--------------------------------------------------------------------------
% Initializing the other variables.
time = 1;           %% iteration time
max_time = 100;     %% iteration times
obj_TWCoKmeans = zeros(1,max_time);   %% store the objective value
%--------------------------------------------------------------------------
%Start the iteration
%--------------------------------------------------------------------------
while 1 && time <= max_time
%tic
fprintf('--------------  The %d Iteration Starts ----------------------\n', time);
%--------------------------------------------------------------------------
%first stage,updating the variable U
% variable D store the Euclidean distance between the data points and 
%cluster centers

save('D.mat','D');
D_average = zeros(data_n,K);

for t =1:T
    D_average = D_average + D{t};
end

D_average = (1/(T-1)) * D_average;

save('D_average.mat','D_average');


for t =1:T  
    U{t} = zeros(data_n,K);
    Cluster_dist =(1- nangda - nangda*(1/(T-1)))*D{t} + nangda * D_average;
    [min_dist,Update_cluster_elem]=min(Cluster_dist,[],2);
    Cluster_elem{t} = Update_cluster_elem;
    for i =1:data_n
        U{t}(i,Cluster_elem{t}(i)) = 1;
    end   
end
save('U.mat','U');
%toc
%--------------------------------------------------------------------------
%Sencond step,updating the cluster center
%tic
U_mix = zeros(data_n,K);

for t =1:T
    U_mix = U_mix + U{t};
end

save('U_mix.mat','U_mix');

for t =1:T
    U_mean{t} = zeros(data_n,K);
end

for t =1:T
     U_mean{t} = (1 - nangda - (nangda /(T-1))) * U{t} + (nangda /(T-1)) * U_mix;
end

save('U_mean.mat','U_mean');
for t =1:T
    for k =1:K               
        center{t}(k,:) = X{t}'* U_mean{t}(:,k) / sum(U_mean{t}(:,k));             
    end
end

save('center.mat','center');
%tic
for t =1:T
    for k = 1:K
       center_sum = repmat(center{t}(k,:),data_n,1);
       D{t}(:,k)  = (X{t} - center_sum).^2*W{t}';              
    end
end
%toc
%--------------------------------------------------------------------------
%Third step, update feature weighting
%tic   
  for t =1:T
       B{t}  =zeros(1,dv(t));
           
  end
  for t = 1:T     
     for i=1:data_n
        for k=1:K     
            if U_mean{t}(i,k) ~=0  
                     B{t} = B{t} + V(t) * U_mean{t}(i,k)*(X{t}(i,:) -center{t}(k,:)).^2;
            end
         end
     end
     
     
     B{t} = exp( (-B{t} - beta) /beta);
 end
         for t =1:T     
             W{t}(1,:) = B{t}./sum(B{t});
         end
 
        save('W.mat','W');
       
        
 %toc
%--------------------------------------------------------------------------

% Fourth step, updating the view.
%tic
V = zeros(1,T);
E = zeros(1,T);

for t = 1:T
   Et = sum(sum(U_mean{t}.*D{t}));            
   E(t) = exp( (-Et - alf) / alf);    
end

save('E.mat','E');

for t =1:T
    V(t) = E(t)/ sum(E);
end
save('V.mat','V');
%toc
%--------------------------------------------------------------------------
%Computing the objective value 

f1 = zeros(T,1);
for t =1:T
    f1(t) = f1(t) + sum(sum(U{t}.* D{t}));          
    f1(t) = f1(t)*V(t)/dv(t);
end
f1_sum = sum(f1);

save('f1_sum.mat','f1_sum');
f2 = zeros(T,1);

for t =1:T
    for tt =1:T                
      f2(t) = f2(t) + sum(sum(abs(V(t)* U_mean{t}.*D{t} - V(tt)*U_mean{tt}.*D{tt})));               
    end
end


f2_sum = sum(f2);
f2_sum = nangda / (T-1)*f2_sum;

save('f2_sum.mat','f2_sum'); 

f3 = 0;

for t =1:T
    if V(t) >= 1e-5
        f3 = f3 + V(t) *log(V(t));
    end
end


f3 = alf * f3;
save('f3.mat','f3'); 

f4 = 0;

for t =1:T    
        f4 = f4 + sum(W{t}.*log(W{t}));
end

f4 = beta * f4;

save('f4.mat','f4'); 
obj_TWCoKmeans(time) = f1_sum + f2_sum + f3 +f4;
%toc
%--------------------------------------------------------------------------
%printf the result 
fprintf('TWCoKmeans: Iteration count = %d, TWCoKmeans = %f\n', time, obj_TWCoKmeans(time));
     
      if  time>1 && (abs(obj_TWCoKmeans(time)-obj_TWCoKmeans(time-1)) <= 1e-5)
          [ index ] = Utoindex( U,data_n,T,K,V);
          fprintf('---------------  The Iteration has finished.-------------------------------\n\n');
          break;
      end
      
 time = time + 1;

end
end

