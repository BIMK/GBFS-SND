function [DEG] = my_degree(Gadj,featSeq)
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
global vWeight;

lastWeg = vWeight(end,:);
% orideg = degree(G);
orideg = sum(logical(Gadj),2);
% Gadj = full(adjacency(G,'weighted'));
oriweg = sum(Gadj, 2);
DEG = round(sqrt(orideg(featSeq).*oriweg(featSeq).*(1*lastWeg(featSeq)')));%�����ڵ㣨������Ȩ��
% DEG0 = round(sqrt(orideg.*oriweg));
% DEGdiff = DEG-DEG0; %debug
% oriweg = sum(Gadj, 2)^2;
% DEG = round((orideg.*oriweg).*(1/3));
end

