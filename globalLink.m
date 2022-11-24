function [dotLink,cmLinkC,cmLinkW] = globalLink(Cmty,NodeMatrix)
%GLOBALWEIGHT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   ���룺���ż��������³�ģ�
%       ���������
%   �����������Ϊ����������ֵΪ���ż��Ȩֵ

% CmtyNum = tabulate(Cmty);
comNum = max(Cmty);


%% ����������������
cmtyWeight = [];%
cmtyCount = [];
for i = 1:1:comNum % ��һ�����ſ�ʼ
    inodeIdx = find(Cmty(:,:)==i);
    
    for j = i+1:1:comNum % ���������ţ����ظ���
        
        jnodeIdx = find(Cmty(:,:)==j);
        outaction = NodeMatrix(jnodeIdx,inodeIdx);
        
        iNodeAllWeight = sum(outaction,1);
        iNodeAllCount = sum(outaction~=0,1);
        
        cmtyWeight = [cmtyWeight, sum(iNodeAllWeight)];
        cmtyCount = [cmtyCount, sum(iNodeAllCount)];
    end

end

%% ȷ�������е���������
nodeweight = zeros(size(Cmty));
nodecount = zeros(size(Cmty));

for i = 1:1:comNum % ��һ�����ſ�ʼ
    inodeIdx = find(Cmty(:,:)==i);
    
    inner = 1:1:comNum;
    inner(inner==i) = [];
    for j = inner % ����������
        
        jnodeIdx = find(Cmty(:,:)==j);
        outaction = NodeMatrix(jnodeIdx,inodeIdx);
        
        iNodeAllWeight = sum(outaction,1);
        iNodeAllCount = sum(outaction~=0,1);

    for k = 1:1:size(inodeIdx,2)
        nodeweight(inodeIdx(k)) = nodeweight(inodeIdx(k))+iNodeAllWeight(k);
        nodecount(inodeIdx(k)) = nodecount(inodeIdx(k))+iNodeAllCount(k);
    end
    
    end
end

dotLink = [Cmty;nodeweight;nodecount];
% ZZ = [];
% for i = unique(GWeight(1,:))
%     e = dotLink(3,dotLink(1,:)==i);
%     ZZ(i,:) = [i,sum(e)];
% end
% ZZ
% squareform(cmtyCount)
% cmLinkW = squareform(cmtyWeight);
% cmLinkC = squareform(cmtyCount);

cmLinkW = cmtyWeight;
cmLinkC = cmtyCount;
end

