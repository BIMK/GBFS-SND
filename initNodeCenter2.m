function [ el_cluster ] = initNodeCenter2( coeff,cluster )
% ���������������ŵ�����
%coeff ΪȨֵ���� clusterΪ���Ż��ֵĽ��
%     [~,column] = size(coeff);
   
    cluster_count = max(cluster); %�����˶��ٸ�����
    el_cluster = zeros(1,cluster_count);%���ÿ�����ŵ�����
    
    for i = 1:cluster_count     %�ӵ�һ�����ſ�ʼ������ÿ��������Ӱ����
%        inter_cluster_node = [] ;%����Щ�ڵ���ͬһ�������ڲ�
%         for j = 1:column
%             if cluster(j) == i
%                 inter_cluster_node=[inter_cluster_node,j];  %�ҵ�����ͬһ�������ڲ��Ľڵ�
%             end
%         end
        inter_cluster_node = find(cluster==i);%�ҵ�����ͬһ�������ڲ��Ľڵ�
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

        el_cluster(i) = trace(X) + 2*sum(sum(triu(W).^2));  %�����ڲ�������������

    end

end

