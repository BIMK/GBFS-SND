function toChangeWeight(kGenWeiArchive, kGenAccArchive,arChive)
% ����ǰk���������ۣ��ı�ڵ�Ȩ��

global vWeight;

% addVal = kGenWeiArchive.*kGenAccArchive;
% % % addVal = sqrt(addVal);
% % a = reshape(addVal',1,size(arChive,1));
% % w0 = a*arChive;
% 
% % addVal = (1-kGenWeiArchive).*kGenAccArchive;
% % addVal = sqrt(addVal);
% a = reshape(addVal',1,size(arChive,1));
% w0 = a*arChive;
% % vWeight = [vWeight;vWeight+w0];
% vWeight = [vWeight+w0];

%% ���Բ����Ǿ��ȵ�����£�ֻ���������ռ��������������Ȩ��
w0 = kGenWeiArchive*arChive;
vWeight = [vWeight+w0];
end

