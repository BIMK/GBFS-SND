function [p] = kshell(Gadj)
%% ��������
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% Gadj ���ڽӾ���

[~,M] = size(Gadj);
featSeq = 1:1:M;
G = graph( Gadj,'upper');
bucket = {};

temp = [];
it = 1;
bucketidx = 1;
while(numnodes(G))

    D = my_degree(G);
%     DD = degree(G);
%     ADD = [D,DD]
    minD = min(D);
    while(true)
        
        feat = find(D==minD); %�ҵ��ȵ���minD�Ľڵ�

        if (~length(feat))
            bucket{bucketidx,1} = temp';
            temp = [];
            bucketidx = bucketidx+1;
            it = it+1;
            break;
        else
            a = featSeq(feat);
            temp = [temp,a];
            featSeq(feat) = [];        
            G = rmnode(G, feat);
            D = my_degree(G);
        end
    end
end

p = bucket;
end