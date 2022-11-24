function [tempkGenArchive] = disEva(arChive)
% �������Ӽ��������롱����
%   �˴���ʾ��ϸ˵��
global data label trIdx;

[M,~] = size(arChive);
tempkGenArchive = zeros(1,M); % ������
tempkGenArchive2 = zeros(1,M); % ������

trlabel = label(trIdx,:);
% temp = zeros(1,size(DM,1));
for j = 1:1:M % arChive�е�ÿһ�������Ӽ�
    featIdx = arChive(j,:);
    D = pdist(data(trIdx,featIdx));
    DM = squareform(D);
    
    farHit = zeros(1,size(DM,1));
    nearMiss = zeros(1,size(DM,1));
    for i = 1:size(DM,1)
        sameClassIdx = trlabel(:,:)==trlabel(i,:);
        otherClassIdx = ~sameClassIdx;

%         sameClassOrder = sort(DM(i,sameClassIdx), 'descend');
%         otherClassOrder  = sort(DM(i,otherClassIdx),'ascend');
%         farHit(i) = sameClassOrder(1);
%         nearMiss(i) = otherClassOrder(1);
        farHit(i) = max(DM(i,sameClassIdx));
        nearMiss(i) = min(DM(i,otherClassIdx));
    end
%     tempkGenArchive(1,j) = mean(farHit-nearMiss) / sum(arChive(j,:));
    len = length(farHit);
    tmpCom = [farHit,nearMiss];
    tmpComNom = mapminmax(tmpCom,0,1);
    farHitNom = tmpComNom(1:len);
    nearMissNom = tmpComNom(len+1:end);
%     tempkGenArchive(1,j) = mean(farHitNom-nearMissNom); %Խ��Խ��
    tempkGenArchive(1,j) = mean(nearMissNom./farHitNom);
    
%     if ~isempty(find((farHitNom-nearMissNom)<=0))
%         error('error!');
end
        
end

% end % function

