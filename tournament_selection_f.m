function [parent_f] = tournament_selection_f...
    (chromosome_f, chromosome_s, V_f,  V_s, featIdx, pool, tour)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

ND1st_s = chromosome_s(chromosome_s(:,end-1)==1, :); % ��������ĵ�һǰ����
ND1st_s_allNum = size(ND1st_s, 1); % ��Ŀ

ND1st_s_Sum = sum(ND1st_s(:,1:V_s), 1); % 1stǰ����ѡ�������ͳ��

mostChooseIdx = ND1st_s_Sum > floor(ND1st_s_allNum/2);%�����������������allNum/2?
if sum(mostChooseIdx)==0
    % �����������������allNum/2, ��ʹ�ñ�׼Ϊ������Ƶ����ߵ�����������һ��
    mostChooseIdx = ND1st_s_Sum > floor(max(ND1st_s_Sum)/2);
end

featIdx_obj = [featIdx,chromosome_f(:,V_f+1:V_f+2)];
featIdx_rank = non_domination_sort_mod(featIdx_obj, 2, size(featIdx,2));

featIdx_temp = featIdx_rank(:,1:V_s);

interSectNum = sum(mostChooseIdx .* featIdx_temp, 2); % ����,�������Ϊֻ�����߶�Ϊ1��ѡ�У�����������Ӧλ�ĳ˻���Ϊ1
allSetNum = sum(featIdx_temp,2);

condition1_temp = interSectNum./allSetNum; % ����ռ������Խ�ࣨ��Խ��
condition2 = featIdx_rank(:,end-1); % ͬһ�����з�֧��Ĳ���,ԽСԽ��
condition1 = zeros(size(condition1_temp));

rank = sort(unique(condition1_temp), 'descend');
for i  = 1:1:size(rank,1)
    condition1(rank(i)==condition1_temp,:) = i;
end
chromosome_f_rank = [chromosome_f,condition1,-condition2]; % condition2ȡ����Ϊ�˵��ú���
f = replace_chromosome(chromosome_f_rank, 2, V_f+2, pool); % M=2 Ŀ����
parent_f = f(:,1:end-2);
end

