function [outFac,LCglobal] = globalCenter(coeff,CmtyNew)
%GLOBALCENTER �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[dotLink,cmLinkC,cmLinkW] = globalLink(CmtyNew, coeff);

cmaw = cmLinkW./cmLinkC;
cmaw(isnan(cmaw)) = 0;
LCglobal = zeros(size(CmtyNew));
cmCenter = NodeCenter2(squareform(cmaw) ,ones(1,max(CmtyNew)));
for i = 1:1:max(CmtyNew)
    LCglobal(1,CmtyNew==i) = cmCenter(1,i);

end
    if size(LCglobal,2)~= size(coeff,2)
        error('����');
    end
    %
    if any(isnan(LCglobal))
        error('ȫ��Ȩ�س���Nan');
% LCglobal(isnan(LCglobal)) = 1;%��������
    end
    %
outFac = ones(1,size(dotLink,2));
Nzero = dotLink(3,:)~=0;
outFac(1,Nzero) = dotLink(2,Nzero)./dotLink(3,Nzero);
outFac(1,CmtyNew==0) = 0;
end

