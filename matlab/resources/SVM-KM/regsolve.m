function [c,d]=rnsolve(K,T,y,lambda)

% USAGE 
% [c,d]=regsolve(K,T,y,lambda)
%
% solve 
% min \sum( (y-f(x)).^2 ) + lambda ||f||^2 
%
% calculates the coefficients of the corresponding
% regularization networks associated to the kernel K
% and the span of T.
% K is a matrix K_ij=K(x_i,x_j) 
% T is a matrix of size M*N where M is the number of spanning
% functions and T_ij= \phi_j(x_i)
%
% f(x)= \sum_i c_i K(x_i,x) + \sum_j d_j \phi(x)
%
% Output
%
% c,d
%
% 24/10/2000 AR
%

regterm=1e-17;

[n1,n2]=size(T);
[n3,n4]=size(K);
if ~isempty(T)
    
    haut=[(K+eye(n3)*lambda) T];
    bas=[T' zeros(n2,n2)];
    mat=[haut;bas];
    matold=mat;
    
    while rcond(mat)<1e-7                 % Badly conditionned matrix
        mat=matold+regterm*eye(size(mat));
        regterm=regterm*10;
        if regterm >1e-5
            warning('Badly Conditionned Matrix in regsolve....');
            break;
        end;
    end;
    ymod=[y;zeros(n2,1)];
    %-----solving--------------------
    cd=mat\ymod;
    c=cd(1:n1);
    d=cd(n1+1:length(cd));
else
    
    haut=[K+eye(n3)*lambda];
    mat=[haut];
    matold=mat;
    
    while rcond(mat)<1e-15                 % Badly conditionned matrix
        mat=matold+regterm*eye(size(mat));
        regterm=regterm*10;
        if regterm >1e-5
            Error('Badly Conditionned Matrix in regsolve....');
        end;
    end;
    c=mat\y;
    d=0;
end;

