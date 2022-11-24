function CmtyNew = fsPaper15(coeff,TV, CmtyOld, DELT, OMEGA)

CmtyNew = CmtyOld;

clusterNotChanged = true;
while (clusterNotChanged)

    LC = NodeCenter2(coeff, CmtyNew); % ������˹���Ķ�
    infFeat = TV.*LC;
%     infFeat = sqrt(infFeat2);
    % Ӱ����infFeat������ֵ����������Ӧ������λ��0��׼����������ɾ��
    finalSet = CmtyNew;
    finalSet(infFeat < DELT) = 0;    

    % û���µ�������Ҫ����һ����������ɾ����˵�����Ų��ٱ仯
    if ~sum(finalSet-CmtyNew)
        clusterNotChanged = false;
        continue;
    end

    % ����ÿ������ɾ��ǰ���������Ŀ
    CmtyNewNum = tabulate(CmtyNew);
    CmtyNewNum(CmtyNewNum(:,1) == 0,:) = [];        

    temp = tabulate(finalSet);
    clusertNum = size(CmtyNewNum,1);
    finalSetNum = cat(2,(1:clusertNum)', zeros(clusertNum,2)); % ����CmtyNewNum����finalSetNum���������ɾ���������ź�ͳ��Ϊ0�����
    finalSetNum(nonzeros(temp(:,1)),:) = temp(temp(:,1)~=0,:);

    % �ҵ�ɾ����������Ŀ������ֵ����������
    CmtyChangeIdx = find(finalSetNum(:,2) > OMEGA);

    % ���ɾ����û��һ�������е�������Ŀ������ֵ��˵�����Ų��ٱ仯
    if isempty(CmtyChangeIdx)
        clusterNotChanged = false;
        continue;
    end

    % ����Ҫ�����ĵ�������ɾ�����������䲻�����κ�����
    CmtyOld = CmtyNew;
    for i = CmtyChangeIdx'
        featChangeIdx = CmtyOld == i;
        CmtyNew(featChangeIdx) = finalSet(featChangeIdx);
    end    

    % �ж��Ƿ���ϴ����Žṹ����ޱ仯�����Խ��ɾ���������ŵ����
%   disp(~sum(CmtyOld-CmtyNew)); % for debug
    if ~sum(CmtyOld-CmtyNew)
        clusterNotChanged = false;
        continue;
    end
end

end