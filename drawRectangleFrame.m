% ��ԭʼͼ��ָ��λ�ü��Ͼ��α߿���
% ԭ������hold on / hold off��rectangle������figure�л�����ο�
% ���룺
% image                 ��Ҫ���б�ǵ�ͼ��
% windowLocation        ���ο�λ�ã����ʽΪ[x,y]����[x����,y����]����λ��Ϊ���������ҵĶ��㣩
% windowSize            ���ο�ߴ磬���ʽΪ[height,width]����[�߶�,���]
%
% �����
% drawRectangleImage    ������һ��ͼƬ������һ��������
%
function [drawRectangleImage] = drawRectangleFrame(image,windowLocation,windowSize)
[row,col] = size(image); % ����ͼ��ߴ�
x = windowLocation(1);%���ο�λ�����꣬���ʽΪ[x,y]
y = windowLocation(2);
height = windowSize(1);%���ο�ߴ磬���ʽΪ[height,width]����[�߶�,���]
width = windowSize(2);
if((x<=row && y<=col)&&(height<=row && width<=col))
    %disp('���ο�Ϸ���');
    figure(1);imshow(image);
    hold on
    drawRectangleImage = rectangle('Position',[y-width,x-height,width,height],'LineWidth',4,'EdgeColor','r');
    hold off
else
    %disp('���ο򲻺Ϸ���');
end