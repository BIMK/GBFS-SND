function [f]  = genetic_operator_s(parent_chromosome, M, V, px,pm)

% global featSetAndMdlMap  featNumMap  featAccMap;
child_chromosome = CXoperate(parent_chromosome(:,1:V),px,pm);
% child_chromosome = mut(child_chromosome(:,1:V),pm); % �����ڽ����ڣ���ģ����б���

% THRESHOLDSET = (bin2dec(num2str(child_chromosome(:,1:V))))./(2^V-1);
% keySet = num2cell(THRESHOLDSET);
% existIdx = isKey(featSetAndMdlMap, keySet);% �Ƿ��Ѿ������Ѿ��������
% 
% % ��ʷ���ֹ���ֱ��ȡ���
% if sum(existIdx)
%     child_chromosome(existIdx,V+1:V+M) = ...
%         [cell2mat(values(featAccMap, keySet(existIdx))),...
%         cell2mat(values(featNumMap, keySet(existIdx)))];
% end
% % �³��ֵĸ��壬����
[N,~] = size(child_chromosome);

for i = 1:N
%     pause(2);
    child_chromosome(i,V+1:V+M) = evaluate_objective_s(M, child_chromosome(i,1:V));
end

f = child_chromosome;
end