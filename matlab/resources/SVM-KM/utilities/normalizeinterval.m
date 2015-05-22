function [xn,mini,maxi,xtn]=normalizeinterval(x,xt,scale,minimum)

% USAGE
%
%  [xn,mini,maxi,xtn]=normalizeinterval(x,xt,scale)
%
% normalize x so that x is in interval [0,1*scale];
% scale is equal to one by default
%
%

if nargin <3
    scale=1;
end;
if nargin <4
    minimum=0;
end;
[nobs,nvar]=size(x);
mini=min(x);
maxi=max(x);
xn=(x-ones(nobs,1)*(mini-minimum))./( ones(nobs,1)*(maxi-mini+minimum)./scale);
xtn=[];
if ~isempty(xt);
    [nobst,nvart]=size(xt);
   xtn=(xt-ones(nobst,1)*(mini-minimum))./( ones(nobst,1)*(maxi-mini+minimum)./scale);
end;