% 给原始图像指定位置加上矩形边框标记
% 原理：利用hold on / hold off和rectangle函数在figure中绘出矩形框
% 输入：
% image                 需要进行标记的图像
% windowLocation        矩形框位置，其格式为[x,y]，即[x坐标,y坐标]（该位置为矩形最靠下最靠右的顶点）
% windowSize            矩形框尺寸，其格式为[height,width]，即[高度,宽度]
%
% 输出：
% drawRectangleImage    并不是一个图片，而是一个？？？
%
function [drawRectangleImage] = drawRectangleFrame(image,windowLocation,windowSize)
[row,col] = size(image); % 输入图像尺寸
x = windowLocation(1);%矩形框位置坐标，其格式为[x,y]
y = windowLocation(2);
height = windowSize(1);%矩形框尺寸，其格式为[height,width]，即[高度,宽度]
width = windowSize(2);
if((x<=row && y<=col)&&(height<=row && width<=col))
    %disp('矩形框合法！');
    figure(1);imshow(image);
    hold on
    drawRectangleImage = rectangle('Position',[y-width,x-height,width,height],'LineWidth',4,'EdgeColor','r');
    hold off
else
    %disp('矩形框不合法！');
end