function [product, T] = Copy_of_js(dataIdx, DELT, OMEGA, RUNS)
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
% zData = zscore(conData);
featNum = size(zData, 2);

THRESHOLD = 0.5; % �б���ֵ
T = zeros(RUNS,1);
for Rtimes = 1:RUNS
    tic;

    row = size(zData,1);    R = randperm(row);
    trIdx = zeros(row, 1);
    trIdx(R(1:round(row*0.7)),:) = 1;    
    teIdx = ~trIdx;
    trIdx = logical(trIdx);
    
    trData = zData(trIdx,:);trLabel = label(trIdx);
    teData = zData(teIdx,:);teLabel = label(teIdx);
    %% ��ͨ�����
    distanceParaAll = {'correlation'};
    totalRes = {'name','adjacency', 'adjMatrix','weightGarph'};

%         [~,TV] = fisherScore(trData,trLabel);
%     TV = 1./(1+exp(-zscore(TV)));
    
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
    TV = 1./(1+exp(-zscore(TV)));
    coeff = totalRes{2,4};

%     R = 0.3;
%     K = round(featNum*R);
%     K = max(CmtyNew)*OMEGA; % ������=������*��С����������
    sfNum = 0;
    sfCount = zeros(1,max(CmtyOld));
%     clusterNotChanged = true;
%     initEng = initNodeCenter2(coeff,CmtyOld);%���ų�ʼ����
%     sfEng = zeros(1,max(CmtyOld));%��ѡ�����������ڵ�����
    cmtyFulled = zeros(1,max(CmtyOld));%�����Ƿ���ѡ��
%     cmtyFulled(initEng==0) = 1;%�����������ֱ��ѡ��
    
    tabu = tabulate(CmtyOld);
    aloneVecCom = tabu(tabu(:,2)==1)';
    cmtyFulled(aloneVecCom) = 1;
    for ai = aloneVecCom
        CmtyNew(CmtyNew==ai) = 0;
    end
    
    
    ratioEng = zeros(1,max(CmtyOld));% ����/ȫ��
    rEhold = 0.5;
    while (~all(cmtyFulled))

%         LC = NodeCenter2(totalRes{2,4}, CmtyNew); % ������˹���Ķ�
        LClocal = NodeCenter2(coeff, CmtyNew); % ������˹���Ķ�
%         LClocal(isnan(LClocal)) = 1; %�����ڵ�;Ӧ��
        [outFac,LCglobal] = globalCenter(coeff,CmtyNew);
        
        infFeat = TV.*(LClocal+outFac.*LCglobal);
%         infFeat = TV.*(LClocal.*outFac.*LCglobal);
%         infFeat = 1./(1+exp(-zscore(infFeat)));

        % �������ŵ�Ӱ������0�������Ͳ���ѡ��
        inftemp = infFeat;
        fulledidx = find(cmtyFulled~=0);
        for r = fulledidx
            inftemp(r==CmtyNew) = 0;
        end
        
        idseq = find(inftemp(:) == max(inftemp(:)));
        if length(idseq)>1
            error('2');
        end
%         id = randperm(length(idseq),1); %���ѡһ�������迼����TV��ߵģ������������?
        id = randperm(length(idseq),1); %��TV��ߵģ������������

        
        updateCmtyIdx = CmtyNew(1,idseq(id));% ѡ����������Ӧ������
        CmtyNew(1,idseq(id)) = 0;%ѡ��idseq(id)������
        sfNum = sfNum+1; %������+1
        sfCount(updateCmtyIdx) = sfCount(updateCmtyIdx)+1;
        % ��Ҫ����������������������
        fIdx = CmtyOld == updateCmtyIdx;
%         sfEng(updateCmtyIdx) = featEng(coeff(fIdx,fIdx),CmtyNew(fIdx));
%         ratioEng(updateCmtyIdx) = 1-sfEng(updateCmtyIdx)/initEng(updateCmtyIdx);
        ratioEng(updateCmtyIdx) = featEng(coeff(fIdx,fIdx),CmtyNew(fIdx));
        if ratioEng(updateCmtyIdx) >= rEhold
            cmtyFulled(updateCmtyIdx) = 1;
        end       


    end

    %  ͳ���������
    selectedFeat.feature = find(CmtyNew==0);
    selectedFeat.num = sum(CmtyNew==0);

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
    
    T(Rtimes,1) = toc;
end
ACC = [ACC; mean(ACC); std(ACC)];
FNUM = [FNUM; mean(FNUM); std(FNUM)];
product = [num2cell(ACC), num2cell(FNUM), FSET];


end % function js