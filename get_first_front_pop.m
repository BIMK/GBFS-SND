function [featIdx]  = get_first_front_pop(individual_adj)
% 返回 featIdx  在将编码转换为sample个体时使用
% global featSetAndMdlMap  featNumMap  featAccMap;
global data trIdx;
global OMEGA DELT;

%% function f = evaluate_objective(x, M, V)
% Function to evaluate the objective functions for the given input vector
% x. x is an array of decision variables and f(1), f(2), etc are the
% objective functions. The algorithm always minimizes the objective
% function hence if you would like to maximize the function then multiply
% the function by negative one. M is the numebr of objective functions and
% V is the number of decision variables. 
%
% This functions is basically written by the user who defines his/her own
% objective function. Make sure that the M and V matches your initial user
% input. Make sure that the 
%
% An example objective function is given below. It has two six decision
% variables are two objective functions.

% f = [];
% %% Objective function one
% % Decision variables are used to form the objective function.
% f(1) = 1 - exp(-4*x(1))*(sin(6*pi*x(1)))^6;
% sum = 0;
% for i = 2 : 6
%     sum = sum + x(i)/4;
% end
% %% Intermediate function
% g_x = 1 + 9*(sum)^(0.25);
% 
% %% Objective function two
% f(2) = g_x*(1 - ((f(1))/(g_x))^2);

%% Kursawe proposed by Frank Kursawe.
% Take a look at the following reference
% A variant of evolution strategies for vector optimization.
% In H. P. Schwefel and R. M鋘ner, editors, Parallel Problem Solving from
% Nature. 1st Workshop, PPSN I, volume 496 of Lecture Notes in Computer 
% Science, pages 193-197, Berlin, Germany, oct 1991. Springer-Verlag. 
%
% Number of objective is two, while it can have arbirtarly many decision
% variables within the range -5 and 5. Common number of variables is 3.
% f = [];
% Objective function one
% sum = 0;
% for i = 1 : V - 1
%     sum = sum - 10*exp(-0.2*sqrt((x(i))^2 + (x(i + 1))^2));
% end
% % Decision variables are used to form the objective function.
% f(1) = sum;
% 
% % Objective function two
% sum = 0;
% for i = 1 : V
%     sum = sum + (abs(x(i))^0.8 + 5*(sin(x(i)))^3);
% end
% % Decision variables are used to form the objective function.
% f(2) = sum;

% corrMatrix =  cutAdjMatrix(RMatrix, THRESHOLD, 0);
corrMatrix = squareform(individual_adj); % 个体代表的网络
if ~sum(any(corrMatrix)) % ==0 说明矩阵所有元素为零
%     f = [inf, inf]; % 上个版本次数可能有误，不应为[-inf, inf]
    featIdx = logical(ones(1,size(data,2)));
else
TV = Term2(data(trIdx,:));

% COMTY = cluster_jl(corrMatrix); % 鲁文
COMTY = cluster_jl_cpp(corrMatrix); % 鲁文
CmtyOld = COMTY.COM{1,end}; % 社团划分的最终输出（鲁文）

CmtyNew = fsPaper15(corrMatrix, TV, CmtyOld, DELT, OMEGA);

featIdx = CmtyNew~=0;

% % 训练集上训练和评价
% mdl = fitcknn(data(trIdx, featIdx), label(trIdx,:), 'NumNeighbors', 5);
% plabel = predict(mdl, data(trIdx, featIdx));
% 
% f(1) = -sum(plabel==label(trIdx,:))/sum(trIdx); %精度,最大化，取负
% f(2) = sum(featIdx); %特征数
end
% 如果是新个体，则加入历史记录中
% if ~isKey(featSetAndMdlMap,THRESHOLD)
%     featSetAndMdlMap(THRESHOLD) = struct('featIdx',featIdx, 'Mdl', mdl);
%     featAccMap(THRESHOLD) = f(1);
%     featNumMap(THRESHOLD) = f(2);
% end

%% Check for error
% if length(f) ~= M
%     error('The number of decision variables does not match you previous input. Kindly check your objective function');
% end