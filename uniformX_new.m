function [NewChrom] = uniformX_new(OldChrom,px,pm)
%UNIFORMX �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[M,N] = size(OldChrom);

parent_1 = 1:M;
parent_2 = randperm(M);

eqIdx = parent_1==parent_2;
while(sum(eqIdx))
    parent_2(eqIdx) = randi([1,M], 1,sum(eqIdx));
    eqIdx = parent_1==parent_2;
end

template_x = logical(randsrc(M,N, [1,0;px,1-px]));

% child_1 = OldChrom(parent_2,:) & template + OldChrom(parent_1,:) & ~template;
% child_2 = OldChrom(parent_1,:) & template + OldChrom(parent_2,:) & ~template;

child_1 = OldChrom(parent_2,:) .* template_x + OldChrom(parent_1,:) .* ~template_x;
child_2 = OldChrom(parent_1,:) .* template_x + OldChrom(parent_2,:) .* ~template_x;

child_1 = mut(child_1, pm); % ����
child_2 = mut(child_2, pm); % ����

NewChrom((1:2:2*M),:) = double(child_1);
NewChrom((2:2:2*M),:) = double(child_2);
% save('111.mat','child_1','template_x','OldChrom','NewChrom')
% ccc
end