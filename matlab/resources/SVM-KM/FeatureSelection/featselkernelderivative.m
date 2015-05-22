function [kernelderiv_1,kernelderiv_2]=featselkernelderivative(k,x_k,kernel,kerneloption,method,x)


%  
%  Process the kernel derivative with regards to variable k
%
%  k is an already process kernel wrt all variables
%  x_k is the data point wrt to variable k
%  kernel is the kind of kernel used 'gaussian' or 'poly'
%  kerneloption 
%  method is either 'grad' or 'scal'
%
%
%  This function computes the derivatives of a kernel with respect to a
%  scaling factor which values is supposed to be equal to 1.



kernelderiv_1=[];
kernelderiv_2=[];
switch kernel
    
case 'gaussian'
    
    switch method
        
    case 'grad'
        nbx=size(x_k,1);
        derivaux=x_k*ones(1,nbx)- ones(nbx,1)*x_k';
        kernelderiv_1= - (derivaux.*k)./kerneloption.^2; % with regards to x_k
        kernelderiv_2=   (derivaux.*k)./kerneloption.^2; % with regards to x'_k       
    case 'scal'
        nbx=size(x_k,1);
        derivaux=(x_k*ones(1,nbx)- ones(nbx,1)*x_k').^2;    
        kernelderiv_1= - (derivaux.*k)./kerneloption.^2;    %with regards to a scaling factor
                        
    end;
    
case 'poly'
    
    switch method    
    case 'grad'
        nbx=size(x_k,1);
        derivaux1=ones(nbx,1)*x_k';
        derivaux2=x_k*ones(1,nbx);
        if kerneloption > 1
            kernelderiv_1= kerneloption*derivaux1*svmkernel(x,kernel,kerneloption-1); % with regards to x_k
            kernelderiv_2= kerneloption*derivaux2*svmkernel(x,kernel,kerneloption-1); % with regards to x'_k
        else
            kernelderiv_1= kerneloption*derivaux1; % with regards to x_k
            kernelderiv_2= kerneloption*derivaux2; % with regards to x'_k
            
        end;
    case 'scal'
        nbx=size(x_k,1);
        derivaux=2*(x_k*ones(1,nbx)).*(ones(nbx,1)*x_k'); 
        if kerneloption >1
            kernelderiv_1= kerneloption*derivaux*svmkernel(x,kernel,kerneloption-1);  %with regards to a scaling factor
        else
            kernelderiv_1=derivaux;
        end;
    end;
    
    
    
otherwise
    
    error('Undefined kernel....')
    
end;
