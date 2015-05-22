function T=waveletspan(x,dilation,uo,a,leftbound,rightbound);

% Usage T=waveletspan(x,dilation,uo,a,leftbound,rightbound)
%
% T is a matrix T_{ij}=\psi_j(x_i)
%   
% x : vector containing samples evaluations
% 
%
% Octobre 2000- AR 

maxi=20;
if nargin<3
   xsup=x;
end;
[n1 n2]=size(x);
%[n3 n4]=size(xsup);
if nargin <2
   uo=0.5;
   a=2;
   leftbound=-1;
   rightbound=11;
   dilation=[0 4];
end;	




nbvectorspan=0;
for dil=dilation;%dilmin:dilmax
   k=round(-leftbound/a^dil/uo):1:round(rightbound/a^(dil)/uo);
   nbvectorspan=nbvectorspan+length(k);
end;
T=zeros(n1,nbvectorspan);
%fprintf('Nb Wavelet Span : %d\n',nbvectorspan);
if nbvectorspan>300
   error('Too many wavelets in span...');
end;
for i=1:n1
   ind=0;
   for dil=dilation;%dilmin:dilmax
      k=round(-leftbound/a^dil/uo):1:round(rightbound/a^(dil)/uo);
      ti= (x(i,:)-k*uo*a^dil)/a^dil;
      %size(ti)
      %pause
      ti=ti.*(abs(ti)<maxi)+maxi.*(ti>=maxi)-maxi.*(ti<=-maxi);
      auxi=2/sqrt(3)*(pi)^(-0.25).*(ti.^2-1).*exp(-ti.^2/2)/sqrt(a^dil);
      size(auxi);
      size(T(i,ind+1:ind+length(ti)));
      T(i,ind+1:ind+length(ti))=auxi;
      ind=ind+length(ti);
   end;
end;


