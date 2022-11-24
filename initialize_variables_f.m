function [f,featIdx_f] = initialize_variables_f(N, M, V, templateAdj)

K = M + V;
% kNeiZoutMode = zeros(size(squareform( templateAdj)));
f = randi([0,1], N, V); % f Ϊ��Ⱥ,V Ϊ����


% f = f.*logical(templateAdj).*templateAdj;

for i = 1 : N
    p = decodeNet(f(i,1:V),templateAdj); %01����ת��Ϊ�����ʾ
    [f(i,V + 1: K),featIdx_f(i,:)] = evaluate_objective_f(M, p);
end