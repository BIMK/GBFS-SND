function [NewChrom] = CXoperate(parent_chromosome,px,pm)
%CXOPERATE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

CXcode = 1;

switch CXcode
    case 0
        NewChrom = uniformX(parent_chromosome,px,pm); % �ɰ汾�������

    case 1
        NewChrom = uniformX_new(parent_chromosome,px,pm);
        
    case 2
        NewChrom = xovmp( parent_chromosome, px, 2, 1 );
        NewChrom = mut(NewChrom, pm);
        
    otherwise
        error("��������");
            
end

