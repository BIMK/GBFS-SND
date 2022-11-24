function [subset] = MAXCG2FS(bucket, Gadj)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
% ��ÿһshell��ѡȡ���������ڵ㣨�����ͨ��ͼ��
%   �˴���ʾ��ϸ˵��
[M2,~] = size(bucket);
M = M2-1;%���һ��ȫѡ,���ô˷������һ����ѡȡ�����ͨ��ͼ�����������һ��
[~,featNum] = size(Gadj);
featSeq = 1:1:featNum;
OriGraph = graph( Gadj,'upper');

binAll = cell(M,1);
binSzAll = cell(M,1); 
cenAll = cell(M,1);

for i = 1:1:M
    kFeat = bucket{i,1};
%     tempnode = setdiff(featSeq,kFeat);
%     kGraph = rmnode(OriGraph, tempnode); % ��ʱ
    kGraph = graph(Gadj(kFeat,kFeat), 'upper');
    
%     plot(kGraph,'Layout','force')
    [bin,binsizes] = conncomp(kGraph); %%��ͨ����
%     idx = binsizes(bin)
    binAll{i,1} = bin;
    binSzAll{i,1} = binsizes;
    
    cen = centrality(kGraph,'degree','Importance',kGraph.Edges.Weight);
    cenAll{i,1} = cen;
end

threshNum = 3; %��ͨ��ͼ��ģ��ֵ
% threshNum = binSzAll{end,1};
for i = 1:1:M
    idx{i,1} = binSzAll{i,1}(binAll{i,1}) >= threshNum;
    idx2(i,1) = any(idx{i,1});
end

tmpset = [];
for i = 1:1:M
    if idx2(i,1) ~= 0
        qiz = find(binSzAll{i,1} >= threshNum);
        for j = binSzAll{i,1}(qiz)
            nodeseq = binSzAll{i,1}(binAll{i,1}) == j;
            nodecen = max(cenAll{i,1}(nodeseq));
            maxCG = find(cenAll{i,1} == nodecen);
            LEN = length(maxCG);
            k = randperm(LEN,1);
            featID = maxCG(k);
        end%for
        tmpset = [tmpset;bucket{i,1}(featID)];
    end% if
    
end

subset = tmpset;
            
end