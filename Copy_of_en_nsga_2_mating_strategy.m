function [chromosome_output] = Copy_of_en_nsga_2_mating_strategy(pop,gen, templateAdj,V_f)

global assiNumInside; % �鿴Archive�б����ĵ�һǰ�����ĸ���
global GAPGEN;
fprintf('--GAP=%d\n',GAPGEN);
%% Simple error checking
% Number of Arguments
% Check for the number of arguments. The two input arguments are necessary
% to run this function.
if nargin < 2
    error('NSGA-II: Please enter the population size and number of generations as input arguments.');
end
% Both the input arguments need to of integer data type
if isnumeric(pop) == 0 || isnumeric(gen) == 0
    error('Both input arguments pop and gen should be integer datatype');
end
% Minimum population size has to be 20 individuals
if pop < 5
    error('Minimum population for running this function is 20');
end
if gen < 5
    error('Minimum number of generations is 5');
end
% Make sure pop and gen are integers
pop = round(pop);
gen = round(gen);
%% Objective Function
% The objective function description contains information about the
% objective function. M is the dimension of the objective space, V is the
% dimension of decision variable space, min_range and max_range are the
% range for the variables in the decision variable space. User has to
% define the objective functions using the decision variables. Make sure to
% edit the function 'evaluate_objective' to suit your needs.
% [M, V, min_range, max_range] = objective_description_function();
global featNum;
global kNeigh;
M = 2; % MĿ����
% V_f = featNum*kNeigh; % ������볤��
% V_f = length(templateAdj);
% V_s = featNum;
%% Initialize the population
% Population is initialized with random values which are within the
% specified range. Each chromosome consists of the decision variables. Also
% the value of the objective functions, rank and crowding distance
% information is also added to the chromosome vector but only the elements
% of the vector which has the decision variables are operated upon to
% perform the genetic operations like corssover and mutation.
% chromosome = initialize_variables(pop, M, V, min_range, max_range);
[chromosome_f,featIdx] = initialize_variables_f(pop, M, V_f,templateAdj); %featIdx ���������縸�������������Ӽ�
% chromosome_s = initialize_variables_s(pop, M, V_s);
%% Sort the initialized population
% Sort the population using non-domination-sort. This returns two columns
% for each individual which are the rank and the crowding distance
% corresponding to their position in the front they belong. At this stage
% the rank and the crowding distance for each chromosome is added to the
% chromosome vector for easy of computation.
chromosome_f = non_domination_sort_mod(chromosome_f, M, V_f);
% chromosome_s = non_domination_sort_mod(chromosome_s, M, V_s);
%% Start the evolution process
% The following are performed in each generation
% * Select the parents which are fit for reproduction
% * Perfrom crossover and Mutation operator on the selected parents
% * Perform Selection from the parents and the offsprings
% * Replace the unfit individuals with the fit individuals to maintain a
%   constant population size.

shareFlag = false; %���������Ƿ��
arChive = [];% �ݴ�ÿ����֧���
kGenWeiArchive = []; % k����ľ������۽��
kGenAccArchive = []; % K����ľ���
% distanceSet = [];
archivePop = pop;
for i = 1 : gen
    % Select the parents
    % Parents are selected for reproduction to generate offspring. The
    % original NSGA-II uses a binary tournament selection based on the
    % crowded-comparision operator. The arguments are 
    % pool - size of the mating pool. It is common to have this to be half the
    %        population size.
    % tour - Tournament size. Original NSGA-II uses a binary tournament
    %        selection, but to see the effect of tournament size this is kept
    %        arbitary, to be choosen by the user.
    pool = round(pop/2);
    tour = 2;
    % Selection process
    % A binary tournament selection is employed in NSGA-II. In a binary
    % tournament selection process two individuals are selected at random
    % and their fitness is compared. The individual with better fitness is
    % selcted as a parent. Tournament selection is carried out until the
    % pool size is filled. Basically a pool size is the number of parents
    % to be selected. The input arguments to the function
    % tournament_selection are chromosome, pool, tour. The function uses
    % only the information from last two elements in the chromosome vector.
    % The last element has the crowding distance information while the
    % penultimate element has the rank information. Selection is based on
    % rank and if individuals with same rank are encountered, crowding
    % distance is compared. A lower rank and higher crowding distance is
    % the selection criteria.
    
    leapGen = 1;
%     gapGen = 5;
    gapGen = GAPGEN;
    
    parent_chromosome_f = tournament_selection(chromosome_f, pool, tour); % ������

    
    px = 0.9;
    pm = 0.01;
    
    [offspring_chromosome_f,featIdx_f] = ...
        genetic_operator_f(parent_chromosome_f, M, V_f, px,pm,templateAdj); % �����Ӵ�
  
    
%     offspring_chromosome_s = ...
%         genetic_operator_s(parent_chromosome_s, M, V_s, px,pm);
    
%     tt = sum(featIdx_f,2); %debug
    % Perfrom crossover and Mutation operator
    % The original NSGA-II algorithm uses Simulated Binary Crossover (SBX) and
    % Polynomial  mutation. Crossover probability pc = 0.9 and mutation
    % probability is pm = 1/n, where n is the number of decision variables.
    % Both real-coded GA and binary-coded GA are implemented in the original
    % algorithm, while in this program only the real-coded GA is considered.
    % The distribution indeices for crossover and mutation operators as mu = 20
    % and mum = 20 respectively.
%     mu = 20;
%     mum = 20;
%     offspring_chromosome = ...
%         genetic_operator(parent_chromosome, ...
%         M, V, mu, mum, min_range, max_range);

    % Intermediate population
    % Intermediate population is the combined population of parents and
    % offsprings of the current generation. The population size is two
    % times the initial population.
    
    [main_pop_f,~] = size(chromosome_f);   
%     [main_pop_s,~] = size(chromosome_s);


%%
    % ��Ϊ����������ÿ����Ⱥֻ�ϲ��Լ�������Ⱥ
        [offspring_pop_f,~] = size(offspring_chromosome_f);
%         [offspring_pop_s,~] = size(offspring_chromosome_s);

        intermediate_chromosome_f(1:main_pop_f,:) = chromosome_f;
        intermediate_chromosome_f(main_pop_f + 1 : main_pop_f + offspring_pop_f,1 : M+V_f) = ...
            offspring_chromosome_f;
        merge_f = intermediate_chromosome_f;
        
%         intermediate_chromosome_s(1:main_pop_s,:) = chromosome_s;
%         intermediate_chromosome_s(main_pop_s + 1 : main_pop_s + offspring_pop_s,1 : M+V_s) = ...
%             offspring_chromosome_s;
%     end
    % Non-domination-sort of intermediate population
    % The intermediate population is sorted again based on non-domination sort
    % before the replacement operator is performed on the intermediate
    % population.
    intermediate_chromosome_f = ...
        non_domination_sort_mod(intermediate_chromosome_f, M, V_f); % ��֧��ֲ�
%     intermediate_chromosome_s = ...
%         non_domination_sort_mod(intermediate_chromosome_s, M, V_s);
    
    % ����ѡ���ĸ����������Ӽ���ֻ�������磩������Ⱥ����ʱ���õ�
    temp_featidx = [featIdx;featIdx_f]; % ��+��
    temp_NDsort = merge_f(:,V_f+1:V_f+M);    
    intermediate_featIdx = [temp_featidx,temp_NDsort];
    
    intermediate_featIdx = non_domination_sort_mod(intermediate_featIdx, M, featNum);    
    
    parentPop = [featIdx,chromosome_f(:,V_f+1:V_f+M)];
    tidx = parentPop(:,end-1)==1;
    t = abs(parentPop(tidx,featNum+1))';
    if size(t,1)>archivePop % ÿ����ౣ��archivePop����Ӣ��
        [~,tidx] = sort(t,'descend');
        tidx = tidx(1:archivePop);
        t = t(tidx);        
    end
    kGenAccArchive = [kGenAccArchive,t]; % ������ǰk����֧���ľ���
    tmpArc = parentPop(tidx,1:featNum);
    arChive = [arChive; tmpArc];% ������ʷ��֧��⣨�����Ӽ���
    tempkGenArchive = disEva(logical(tmpArc));% �������۵�ǰ����֧���
    kGenWeiArchive = [kGenWeiArchive,tempkGenArchive]; % k������ϲ�������
    
%     arChive = []; % �ⲿ�ĵ������Ž���գ�׼�������´����Ž�
    tempkGenArchive = []; % ���
    
    featIdx = replace_chromosome(intermediate_featIdx, M, featNum, pop);
%     save('C:\Users\BIMK\Desktop\SHIX\0000021.mat','intermediate_featIdx','featIdx');
    
    featIdx = logical(featIdx(:,1:featNum));
    % Perform Selection
    % Once the intermediate population is sorted only the best solution is
    % selected based on it rank and crowding distance. Each front is filled in
    % ascending order until the addition of population size is reached. The
    % last front is included in the population based on the individuals with
    % least crowding distance
    chromosome_f = replace_chromosome(intermediate_chromosome_f, M, V_f, pop); % ǰ�����滻��һ��
%     chromosome_s = replace_chromosome(intermediate_chromosome_s, M, V_s, pop);    



%   �ӵ�K����ʼ,������ 
 
    if i>=leapGen, shareFlag = true; end
    
    if shareFlag && ~mod(i, gapGen)   % ÿ��K���仯�ڵ�Ȩ��

        toChangeWeight(kGenWeiArchive, kGenAccArchive,arChive);% �ı�Ȩ�صĲ��� % 
        assiNumInside = [assiNumInside;size(kGenWeiArchive,2)];
        kGenWeiArchive = []; % ǰk�������ĵ�һǰ���������Ϣ�ÿ�
        kGenAccArchive = [];
        arChive = [];
    end


    

end

% chromosome_s2f = s_2_f(chromosome_s(:,1:V_s),M);
% chromosome_all = [chromosome_s2f;chromosome_f(:,1:V_f+M)]; % ���շ��صĸ�������������

chromosome_f2s = [featIdx,chromosome_f(:,V_f+1:V_f+M)];
% chromosome_s_convert = [chromosome_s(:,1:V_s),chromosome_s(:,V_s+M), sum(chromosome_s(:,1:V_s),2)];
% chromosome_all = [chromosome_s_convert;chromosome_f2s];
% chromosome_all = chromosome_f2s;
% intermediate_chromosome = non_domination_sort_mod(chromosome_all, M, V_s);
% chromosome_output= replace_chromosome(intermediate_chromosome, M, V_s, pop);

chromosome_output = chromosome_f2s;
%% Result
% Save the result in ASCII text format.
% save solution.txt chromosome -ASCII

%% Visualize
% The following is used to visualize the result if objective space
% dimension is visualizable.
% if M == 2
%     plot(chromosome(:,V + 1),chromosome(:,V + 2),'*');
% elseif M ==3
%     plot3(chromosome(:,V + 1),chromosome(:,V + 2),chromosome(:,V + 3),'*');
% end

% NDkey = num2cell(bin2dec(num2str(chromosome(:,1:V)))./(2^V-1));
% archive = containers.Map(NDkey, values(featSetAndMdlMap,NDkey));
end
