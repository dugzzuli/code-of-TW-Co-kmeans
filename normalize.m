function [ data_new ] = normalize( data )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
   n =size(data,2);
   for i =1:n
      data_new(:,i)= mapminmax(data(:,i)',0,1);
   end

end

