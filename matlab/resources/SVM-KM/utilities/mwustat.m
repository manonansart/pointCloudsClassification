function [u,z] = mwustat(data1,data2)

% [u,z] = mwustat(data1,data2)
%
% mwustat.m: given vectors data1 and data2, the Mann-Whitney U statistic is
% returned along with the significance statistic Z.  The Z value should be
% compared to T(alpha,infinity) for samples larger than 40 data points, 
% otherwise The U value should be compared to a table of the U distribution. 
% This implementation assumes a precision in measurements that renders
% tie values (i.e., > 1 occurence of the same value) unlikely.  If tie values
% are expected, this function should be modified.
%
% Adapted from Biostatistical Analysis, Zar JH, Prentice-Hall, 1974.

data1 = sort(data1);
data2 = sort(data2);
n1 = length(data1);
n2 = length(data2);
rank1 = zeros(n1,1);
rank2 = zeros(n2,1);
i1 = 1;
i2 = 1;
rank = 1;

while (i1 <= n1) | (i2 <= n2),
  if i1 <= n1,
    d1 = data1(i1);
  end
  if i2 <= n2,
    d2 = data2(i2);
  end
  if ((d1 <= d2) | (i2 == n2+1)) & (i1 ~= n1+1),
    rank1(i1) = rank;
    i1 = i1+1;
    rank = rank+1;
  end 
  if ((d2 <= d1) | (i1 == n1+1)) & (i2 ~= n2+1),
    rank2(i2) = rank;
    i2 = i2+1;
    rank = rank+1;
  end 
end

u = n1*n2+((n1*(n1+1))/2)-sum(rank1);
up = n1*n2-u;
u = min(u,up);
z = abs(u-(n1*n2)/2)/sqrt((n1*n2*(n1+n2+1))/12);


