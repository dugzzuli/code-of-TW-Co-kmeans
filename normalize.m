function [ data_new ] = normalize( data )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
   n =size(data,2);
   for i =1:n
      data_new(:,i)= mapminmax(data(:,i)',0,1);
   end

end

