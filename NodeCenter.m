function [ node_center_result ] = NodeCenter( coeff,cluster )
% ����ÿ���ڵ��Ӱ�����������׼Ϊ�򵥵����Լ������Ķȱ��������������ŵĶ�
%coeff ΪȨֵ���� clusterΪ���Ż��ֵĽ��
    [~,column] = size(coeff);
    node_center_result = zeros(1,column);%���ÿ���ڵ�����Ķ�
    
    cluster_count = max(cluster); %�����˶��ٸ�����
    for i = 1:cluster_count     %�ӵ�һ�����ſ�ʼ������ÿ��������Ӱ����
       inter_cluster_node = [] ;%����Щ�ڵ���ͬһ�������ڲ�
        for j = 1:column
            if cluster(j) == i
                inter_cluster_node=[inter_cluster_node,j];  %�ҵ�����ͬһ�������ڲ��Ľڵ�
            end
        end
        [~,w_column]=size(inter_cluster_node);  %�������м����㣬����һ������W �������еģ�2��W��
        W = zeros(w_column,w_column);
        X = W ;           %XΪ�Խ���Ϊ�ܶȾ���  �����е�(3)X
        for w_i = 1:w_column
            for w_j = 1:w_column
                W(w_i,w_j) = coeff(inter_cluster_node(w_i),inter_cluster_node(w_j));  %����ͬһ�������ڲ�������֮�����ϵ��
            end
        end
        
        for w_i = 1:w_column
            X(w_i,w_i) = sum(W(w_i,1:w_column)); %�����ܵĶȾ���XΪ�Խ���
        end

        %����
        el_cluster = sum(X) + sum(W.^2);  %�����ڲ�����������
        candidate = zeros(1,w_column);   %��ź�ѡɾ���ڵ���������ŵ�Ӱ��
        for w_i = 1:w_column             %�������ɾ��ĳ���ڵ������ŵ�Ӱ�� 
            delete_W = W ;
            delete_X = X ;
            delete_W (:,[w_i]) = [] ;
            delete_W ([w_i],:) = [] ;
            delete_X (:,[w_i]) = [] ;
            delete_X ([w_i],:) = [] ;
            
            el_dellete_node = sum(delete_X)+sum(delete_W.^2);
            candidate(w_i) = (sum(el_cluster)-sum(el_dellete_node))/sum(el_cluster) ;
            
        end
        %�˴��Ƿ�Ӧ�����¹�һ���������������Ѿ���һ����
        candidate = 1./(1+exp(-zscore(candidate)));
        for result =1:w_column   %�ҵ����������нڵ�����Ķ�
            node_center_result(inter_cluster_node(result)) = candidate(result);
        end
    end

end

