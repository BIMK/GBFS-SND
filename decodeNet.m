function [indivNet] = decodeNet(f,adj)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
global kNeigh;
global featNum;
global kNeiMatrix;
global kNeiZout

global TTT
% tic
K = kNeiZout;%squareform(adj);
MODE = zeros(size(K));

rShapeF = reshape(f, featNum, kNeigh);
STD = kNeiMatrix.*rShapeF;
for i = 1:1:featNum
%     save('2.mat','STD','featNum')
%     cccc
    idx = find(STD(i,:));
    MODE(i,STD(i,idx)) = 1;
end

indivNet = MODE.*K;
% TT=toc
% TTT(end+1)=TT;
end

