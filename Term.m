function [ term ] = Term( train_set )
    [data_row,data_column] = size(train_set);
         term = zeros(1,data_column);        %ÿ�����������
         for i = 1:data_column
                a =sum(train_set(:,i)) ;
                feature_mean = mean(train_set(:,i));
            %sum / ��
                term(i) = sum((train_set(:,i)-feature_mean).^2); %����ÿһλ�������ֵ�ķ�������ͣ�����һ��ķ���
                term(i) = term(i)/data_row ;   %�������
         end
end

