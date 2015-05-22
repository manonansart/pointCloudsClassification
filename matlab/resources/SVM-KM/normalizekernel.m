function K=normalizekernel(x,kernel,kerneloption,xsup);


%  K=normalizekernel(x,kernel,kerneloption,xsup);
n=size(x,1);

kdiag1=diag(svmkernel(x,kernel,kerneloption));

if nargin >3
 
    kdiag2=diag(svmkernel(xsup,kernel,kerneloption));
    Kp=svmkernel(x,kernel,kerneloption,xsup);
else
    kdiag2=kdiag1;
        Kp=svmkernel(x,kernel,kerneloption);
end;

Kweight=kdiag1*kdiag2';
K=Kp./sqrt(Kweight);