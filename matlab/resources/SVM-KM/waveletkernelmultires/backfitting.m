function C=backfitting(y,kernelmatrix,lambda,MaxIter,verbose)

% USAGE
%
% C=backfitting(y,kernelmatrix,lambda,MaxIter,verbose)
% 
% 

if nargin < 5
    verbose=0;
end
if nargin<4;
    MaxIter=200;
end;
if nargin<2
    error('Not enough input variables');
end;

[n,n1,nbspace]=size(kernelmatrix);

C=zeros(n,nbspace); % matrices des coefficients
Cold=ones(n,nbspace);
iter=0;
%keyboard
while norm(C-Cold,inf)>1e-6 & iter<MaxIter
    
 
    
    Cold=C;
  
    for i=1:nbspace
        
        yaux=y;
        for k=1:nbspace
            if k~=i
                yaux=yaux-kernelmatrix(:,:,k)*C(:,k);
            end;
        end;
        C(:,i)=regsolve(kernelmatrix(:,:,i),[],yaux,lambda(i));
    end;  
% close all
%    plot(kernelmatrix(:,:,1)*C(:,1),'r'),hold on, plot(kernelmatrix(:,:,2)*C(:,2),'b')
   
    iter=iter+1; 
    if verbose
    fprintf('%d..',iter);
    end;
end;
if verbose
fprintf('\n');
end;