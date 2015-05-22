clear all
close all
clc

%% Bounding box
nuage = loadpcd('../../dataset/lomita/track_068segment_005.pcd');
x=nuage(1,:);
y=nuage(2,:);
z=nuage(3,:);
x=double(x'); % transformation en double et transposée de chaque vecteur pour pouvoir utiliser minboundbox
y=double(y');
z=double(z');
figure
plot3(x,y,z, '+');


[rotmat,cornerpoints,volume,surface] = minboundbox(x,y,z);

figure
plot3(x,y,z,'b.');hold on;plotminbox(cornerpoints,'r');

 
%% Output de minboundbox : 
% rotmat - (3x3) rotation matrix for mapping of the pointcloud into a
                  % box which is axis-parallel (use inv(rotmat) for inverse
                  % mapping).
%  
 % cornerpoints - (8x3) the cornerpoints of the bounding box.
%  
 % volume - (scalar) volume of the minimal box itself.
%  
 % surface - (scalar) surface of the minimal box as found.
%  
 % edgelength - (scalar) sum of the edgelengths of the minimal box as found.| 