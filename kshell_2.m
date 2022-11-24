function [p] = kshell_2(Gadj0)
%UNTITLED2 �Ľ���k-shell ����������ȡ�����Ӽ�
%   �ο������k-shell �㷨


% Gadj ���ڽӾ���
% global featMAXCG;
Gadj = Gadj0; %����
[~,M] = size(Gadj);
featSeq = 1:1:M;
% G = graph( Gadj,'upper');
bucket = {};

temp = [];
it = 1;
bucketidx = 1;
while(~isempty(featSeq)) % ѭ��ֱ�����е㱻ɾ��

    D = my_degree(Gadj,featSeq);
%     DD = degree(G);
%     ADD = [D,DD]
    minD = min(D);
    if sum(isnan(D))
        error('�������Nan');
    end
    while(true)
        
        feat = find(D==minD); %�ҵ��ȵ���minD�Ľڵ�

        if (~length(feat))
            bucket{bucketidx,1} = temp';
            temp = [];
            bucketidx = bucketidx+1;
            it = it+1;
            break; %����
        else
            a = featSeq(feat);
            temp = [temp,a];
            featSeq(feat) = [];        
%             G = rmnode(G, feat); % ��ʱ
%             G = graph( Gadj(featSeq,featSeq),'upper'); % ͬ���ܺ�ʱ
            Gadj(feat,:) = 0;
            Gadj(:,feat) = 0; %��Ӧ������������ߵ�Ȩ����0����ͬɾ��
            D = my_degree(Gadj,featSeq);
        end
    end
end



% 
% feat =MAXCG2FS(bucket,Gadj);


[Nbkt,~] = size(bucket);
if Nbkt>1
    feat=select(bucket);
    p = [feat,bucket{Nbkt,1}'];
else
    p = bucket{Nbkt,1}';
end

p = sort(p);

end