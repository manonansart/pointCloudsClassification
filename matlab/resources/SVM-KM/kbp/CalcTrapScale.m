function [sigma] = CalcTrapScale(xapp)

%
% Calcul l'echelle piege pour le LARS
% USE : [sigma] = CalcTrapScale(xapp)
%

% Hypothese : le bruit est de l'ordre de la frequence d'echantillonnage ???

% trouver la plus petite distance entre deux points :

[distance] = calcdistance(xapp);
eps        = max(diag(distance));

ind        = find(distance>eps);
min_dist   = min(distance(ind));

% min_dist   = xapp(2) - xapp(1);
sigma      = sqrt(-min_dist^2/(2*log(0.001)));

%keyboard



% kerneloption = [sigma*20 sigma*15 sigma*10 sigma*5 sigma];