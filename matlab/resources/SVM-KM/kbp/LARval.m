function ypred = LARval(K,solution)

% USAGE
%
% ypred = LARval(K,solution) 

if size(K,2)==size(solution.Beta,2);
    ypred                 = K    *solution.Beta'+solution.b;    
else
    ypred                 = K (:,solution.indxsup)   *solution.Beta'+solution.b;    
end;