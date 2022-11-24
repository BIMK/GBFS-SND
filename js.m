function product = js(dataIdx, DELT, OMEGA, RUNS)
%% Ԥ����
% clear;
% clc;
% data = rand(10,5);
% dbstop if error;
% clear;
%% ���ݵ���
% global zData disData label;
% conData = csvread('dataR2.csv',1);
% disData = csvread('data.csv',1);
% label = conData(:,end);
% datasetName = '����';
% conData = conData(:,1:end-1);
% disData = disData(:,1:end-1);
% % disData = conData;

% DELT = 0.2; % Ӱ������ֵ
% OMEGA = 2; % ������С������Ŀ
% dataIdx = 27; %[27,30,28,12,22,2,31,32]
[conData, label, datasetName] = inputdatasetXD(dataIdx);
disData = [];
[ACC, FNUM] = deal(zeros(RUNS, 1));
FSET = cell(RUNS+2,1);

%% �������� ���ݻ���
zData = mapminmax(conData',0,1)'; % ʹ�øú���ע��ת��
featNum = size(zData, 2);

THRESHOLD = 0.5; % �б���ֵ

for Rtimes = 1:RUNS

    row = size(zData,1);    R = randperm(row);
    trIdx = zeros(row, 1);
    trIdx(R(1:round(row*0.7)),:) = 1;    
    teIdx = ~trIdx;
    trIdx = logical(trIdx);
    
    trData = zData(trIdx,:);trLabel = label(trIdx);
    teData = zData(teIdx,:);teLabel = label(teIdx);
    %% ��ͨ�����
    distanceParaAll = {'correlation','spearman'};
    totalRes = {'name','adjacency', 'adjMatrix','weightGarph'};

    for i = 1:size(distanceParaAll,2)
        adj = 1-pdist(trData', distanceParaAll{1,i}); % ת���Լ������������    
        % 15���Ȩֵ�������˹�һ��
        adj = abs(adj);
        adj = 1./(1+exp(-zscore(adj)));   
        Zout = squareform( adj);

        totalRes = [totalRes; {distanceParaAll{1,i},adj, Zout, cutAdjMatrix(Zout,THRESHOLD, 0)}];
    %     netplot(totalRes{i+1,2},1);
    end

    %% ���ż��
    N = size(totalRes,2);
    for i = 2:size(totalRes,1)
        COMTY = cluster_jl(totalRes{i,4}); % ³��
        totalRes{i,N+2} = COMTY;
        totalRes{i,N+1} = graph(totalRes{i,4});
    end
    totalRes{1,N+1} = 'globalGraph';
    totalRes{1,N+2} = 'community';
    N = N+2;
    %% ����15�����
    % DELT = 0.2; % Ӱ������ֵ
    % OMEGA = 2; % ������С������Ŀ

    CmtyOld = totalRes{2,6}.COM{1,end}; % ���Ż��ֵ����������³�ģ�
    CmtyNew = CmtyOld;

%     TV = Term2(conData(CVO.training,:)); % ���
%     TV = Term2(trData); % ѵ�����ϵ����
    [~,TV] = fisherScore(trData,trLabel);
%     TV = 1./(1+exp(-zscore(TV)));
    coeff = totalRes{2,4};

    clusterNotChanged = true;
    while (clusterNotChanged)

%         LC = NodeCenter2(totalRes{2,4}, CmtyNew); % ������˹���Ķ�
        LClocal = NodeCenter2(coeff, CmtyNew); % ������˹���Ķ�
        LClocal(isnan(LClocal)) = 1;
        [outFac,LCglobal] = globalCenter(coeff,CmtyNew);
        
        infFeat = TV.*(LClocal+outFac.*LCglobal);
        infFeat = 1./(1+exp(-zscore(infFeat)));
%         infFeat = TV.*LC;
    %     infFeat = sqrt(infFeat2);
        % Ӱ����infFeat������ֵ����������Ӧ������λ��0��׼����������ɾ��
        finalSet = CmtyNew;
        finalSet(infFeat < 0.5) = 0;    

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
        CmtyChangeIdx = find(finalSetNum(:,2) >= OMEGA);

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

    %  ͳ���������
    selectedFeat.feature = find(CmtyNew~=0);
    selectedFeat.num = sum(CmtyNew~=0);

    totalRes{1,N+1} = 'selectedFeat';
    totalRes{2,N+1} = selectedFeat;
    N = N+1;

    %% ����ѡ�� ����
    % ֻʹ����������

    knnModel = fitcknn(trData(:,totalRes{2,N}.feature), trLabel, 'NumNeighbors', 5);
    predLabel = predict(knnModel, teData(:,totalRes{2,N}.feature));
    totalRes{1,N+1} = 'CMTYFitness';
    acc = sum(predLabel==teLabel)/length(teLabel);
    totalRes{2,N+1} = acc;
    N = N+1;
    %% ��ͼ
    % for i = 2:size(totalRes,1)
    %     figure('Name',totalRes{i,1});
    % %     G = graph(totalRes{i,4});
    % %     plot(G,'Layout','force','EdgeLabel',G.Edges.Weight);
    %     plot(totalRes{i,5},'Layout','force');
    %     title(totalRes{i,1});
    % end

    %% ���
    ACC(Rtimes) = acc;
    FNUM(Rtimes) = totalRes{2,7}.num;
    FSET{Rtimes} = totalRes{2,7}.feature;
end
ACC = [ACC; mean(ACC); std(ACC)];
FNUM = [FNUM; mean(FNUM); std(FNUM)];
product = [num2cell(ACC), num2cell(FNUM), FSET];


end % function js