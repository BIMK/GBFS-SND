function [TV] = Term2(dataset)
%TERM2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[M,~] = size(dataset);
fAve = mean(dataset, 1);
TV = sum((dataset-fAve).^2)./M;
% ��һ��
TV = 1./(1+exp(-zscore(TV)));

end

