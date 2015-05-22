function K=normalizekernel(x,kernel,kerneloption,xsup);

%  USAGE
%
% K=normalizekernel(x,kernel,kerneloption,xsup);

n=size(x,1);
for i=1:n
    kdiag1(i,:)=svmkernel(x(i,:),kernel,kerneloption);
end;
if nargin >3
    n1=size(x1,1);
    for i=1:n1
    kdiag2(i,:)=svmkernel(xsup(i,:),kernel,kerneloption);
    end;
else
    kdiag2=kdiag1;
end;
Kp=svmkernel(x,kernel,kerneloption,xsup);
Kweight=kdiag1*kdiag2';
K=Kp./sqrt(Kweight);