function [Tp,Tm,z,p] = wstat(data1,data2)

% Usage [Tp,Tm,z,p] = wstat(data1,data2)
% given vectors data1 and data2, the Wilcoxon paired-sample test statistics, 
% T+ and T-, are returned.  
% 
% z : z-value of Wilcoxon paired-sample (Value of the nonparametric test)
% p : p- value 
%
% 
%  H0 : values in data1 vector are "equivalent" to those in data2 
%       mean value in data1 are equal to mean value in data2
%  H1 : values  in data1  are "higher" than those in data2
%
%
%
% Available as the signrank function in the Statistics toolbox
% data1 and data2 must be column vector


%  Modified AR
% Adapted from Biostatistical Analysis, Zar JH, Prentice-Hall, 1999, pp. 165-167.


if nargin ~= 2,
   error('Wstat requires two input arguments.');
elseif length(data1) ~= length(data2),
   error('The input arguments for wstat must be the same size.');
end


i1 = 1;
i2 = 2;
rsum = 0;
rank = 1;

a = [data1 - data2, abs(data1 - data2)];
ind=find(a(:,2)~=0);
a=a(ind,:);
b = sortrows(a,2);
d = b(:,2);
n = size(a,1);
if n<=1
    warning('data1 and data2 have to many ties or are too small...');
    z=5;
    Tp=0;
    Tm=0;
    return
end;
while i1 <= n,
   while (i2 <= n) & (d(i1) == d(i2)), % in case of ties
      i2 = i2 + 1;
   end
   ranks(i1:i2-1) = sum(rank:rank + (i2 - 1 - i1)) / (i2 - i1);
   rank = rank + (i2 - i1);
   i1 = i2;
   i2 = i1 + 1;
end
Tp=sum(ranks(find(b(:,1) >= 0)));
Tm=sum(ranks(find(b(:,1) < 0)));
      
W=Tp-Tm;
sigmaw=sqrt(n*(n+1)*(2*n+1)/6);
z=(W-0.5)/sigmaw;
p=normcdf(z);

