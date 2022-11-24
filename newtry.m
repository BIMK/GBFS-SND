function [bestAdj] = newtry(inputAdj,THRESHOLD, pop, times)
%NEWTRY 
%�ýϵ͵���ֵ�з����磬ȥ��һ��������Խϵ͵ıߣ�������
%Ȼ����NSGA2����ڵ㣬�ҵ�һ�����ŵ��������磬���������Ӽ�
%   �˴���ʾ��ϸ˵��
if nargin < 3
    pop = 20;
    times = 40;
end

templateAdj = inputAdj;
templateAdj(templateAdj<THRESHOLD) = 0;

% chromes = en_nsga_2(pop,times,templateAdj);  % ˫��Ⱥ
chromes = en_nsga_2_mating_strategy(pop,times,templateAdj);

[~,n] = size(squareform(chromes(1,1:end-4)));


alpha = 0.6;
fits = alpha.*abs(chromes(:,end-3))+...
    (1-alpha).*(1-chromes(:,end-2)./n);
selected = find(fits==max(fits));
idx = selected(1,:);
bestAdj = chromes(idx,1:end-4);

% sprintf('ѡ�����ѣ�%.2f  %d', abs(chromes(idx,end-3)), chromes(idx,end-2))
end

