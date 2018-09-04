% Numerical Methods for Deep Learning
% 
% Example for color based segmentation.
%
% Requires:
%   coloredChips.png  - image data
%   imgSeg.mat        - an (imperfect) segmentation

clear 
close all

vec = @(x) x(:);

I = imread('coloredChips.png');
I = I(1:300,101:450,:);

figure(1); clf;
subplot(1,3,1)
imagesc(I);
axis off;
title('input image');

pause

load imgSeg
subplot(1,3,2)
imagesc(imgSeg);
axis off;
title('segmentation')

pause

RGB = [vec(I(:,:,1)), vec(I(:,:,2)), vec(I(:,:,3))];

i1 = imgSeg==1;
i2 = imgSeg==2;
i3 = imgSeg==3;

subplot(1,3,3);
plot3(RGB(i1,1),RGB(i1,2),RGB(i1,3),'.b')
hold on
plot3(RGB(i2,1),RGB(i2,2),RGB(i2,3),'.r')
hold on
plot3(RGB(i3,1),RGB(i3,2),RGB(i3,3),'.y')
hold off
title('RGB values as 3D coordinates');

