function [featidx] = newtry_ms(inputAdj,THRESHOLD, pop, times)
%NEWTRY 
%��k���ڷ���������
%Ȼ����NSGA2����ڵ㣬�ҵ�һ�����ŵ��������磬���������Ӽ�
global trData trLabel teData teLabel;
global  featNum kNeigh;


V_f = featNum*kNeigh;
if nargin < 3
    pop = 20;
    times = 40;
end

templateAdj = inputAdj;
% templateAdj(templateAdj<THRESHOLD) = 0;

% chromes = en_nsga_2(pop,times,templateAdj);  % ˫��Ⱥ
% chromes = en_nsga_2_mating_strategy(pop,times,templateAdj); % һ�뾫��
chromes = Copy_of_en_nsga_2_mating_strategy(pop,times,templateAdj,V_f); % ǰ����

[m,n] = size(chromes(:,1:end-2));


 % ȡ�����
alpha = 0.9; % ����Ȩ�أ�Ϊ0��ʾ�����Ǿ��ȣ�1��ʾֻ���Ǿ���
fits = alpha.*abs(chromes(:,end-1))+...
    (1-alpha).*(1-chromes(:,end)./n);
selected = find(fits==max(fits));
idx = selected(1,:);
featidx = chromes(idx,1:end-2);

% feature = logical(chromes(:,1:end-2));
% for i = 1:1:m
%     knnModel = fitcknn(trData(:,feature(i,:)), trLabel, 'NumNeighbors', 5);
%     predLabel = predict(knnModel, teData(:,feature(i,:)));
%     acc(i,:) = sum(predLabel==teLabel)/length(teLabel);
% end
% check = [chromes(:,end-1:end),acc];
% sprintf('ѡ�����ѣ�%.2f  %d', abs(chromes(idx,end-3)), chromes(idx,end-2))
end
