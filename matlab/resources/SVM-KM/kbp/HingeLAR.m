function [solution, solution_OLS] = HingeLAR(x,y, AllBorne, Limites, lambda, verbose, xappforplot)

% Usage
% [solution, solution_OLS] = HingeLAR(x,y, AllBorne, Limites, lambda, verbose, xappforplot)
%
%
%  Looks for a sparse squared hinge loss classifier
%  by solving : min \sum max(0, 1-y f(x)) ^2 
%               sum(abs(Beta)) < t
%
%  The algorithm is based on the LARS of Efron et al and Rosset's paper
%  Stopping criterion and multiple kernel formulation are based on
%  V. Guigue's paper untitled Kernel Basis Pursuit (google it!).
%
% ARGUMENTS :
%   verbose    : plot (>0) or no (==0) lfigures.
%               figures are numbered from verbose
%   x          : input data
%   y          : output data
%   AllBorne :  This paraameter is a structure that deals with the stopping
%               criterion. One can combine several kinds of stopping
%               criterion for the processing of the regularization path.
%
%                    Allborne{i}.type : type of stopping criterion 
%                    Allborne{i}.borne : bound value on this stopping
%                                       criterion
%                                     
%                   Here we give the possible type of stopping criterion
%
%                   'sumBeta'  ->  sum(abs(Beta)) < borne_Beta
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.
%                       
%                       (usual bound as defined in the LARS algorithm)
%
%                   'pcSV'     ->  borne_Beta = percent of selected support
%                               vectors
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.

%                   'nbSV'     ->  borne_Beta = nb of support vector
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.

%                   'trapscale'-> scale or kernel representing noise 
%                       * structure.indTS : index of the kernel trapscale
%                       in the multiple kernel setting. It should be one of
%                       the last.                   
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds. (counts
%                       the number of time the trapscale has been
%                       selected.)
%
%                   'RVtrap'   ->  Automatically add some random noise as
%                   information sources.
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds. (counts
%                       the number of time the RV has been selected).
%
%                   'countback' -> allowed backward movement
%                       *  Allborne.borne : vector of bounds. The algorithm
%                       will output as many solutions as bounds.
%
%   Limites         * structure.nbmaxMoveBack : maximal number of backward
%                     movement for the LASSO.
%                   * structure.nbmaxSV : maximal number of SV allowed.
%                       this is useful in a Large Scale problem in order to
%                       avoid memory crash.
%
%   lambda : regularisation term
%
% OUTPUT:
%   solution  : all the solutions in a structures, the beta are only the
%                   non-zeros one
%                   solution{i}.b   
%                   solution{i}.Beta
%                   solution{i}.indxsup
%                   solution{i}.type % de solution !
%                   solution{i}.borne
%
%   solution_OLS : same solution  but the last step is based on the OLS not
%   on LARS.
%
%


% A. Rakotomamonjy 01/2007
% V. Guigue 05/2006


if nargin<6
    verbose = 0;
end
if nargin<5
    lambda = 1e-6;
end
if nargin <4
    Limites = [];
end

test_OLS = 1;

nbvar    =  size(x,2);
nbptsapp =  size(x,1);
if sum(std(x)) > nbvar+0.5 | sum(std(x)) < nbvar-0.5
    %error('LARS method should be used on normalized data\n');   
    fprintf('LARS method should be used on normalized data\n');
end

nbexptot = 0;
compt_backward = 0;
nbborne  = length(AllBorne);
for i=1:nbborne,
    nbexptot = nbexptot + length(AllBorne{i}.borne);
    
end
nbexpcur = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AJOUT DE B NORMALISE

x = [x ones(size(x,1),1)/sqrt(nbptsapp)];


if isfield(Limites,'nbmaxMoveBack') == 0
    Limites.nbmaxMoveBack = 1e8;
end
if isfield(Limites,'nbmaxSV') == 0
    Limites.nbmaxSV = 1e8;
end


% le LAR
% initialisation  : b = 0;
f = zeros(nbptsapp,1);
A = [];
Abar = 1:size(x,2); % le biais et les multiK sont pris en compte
b = [];

Auto_Ajout_PTS = 1;
iteration = 0;
TT=1;
anc_max_cor=inf;
%
mem_cor = [];
mem_beta = [];
mem_cout = [];
arret = 0;
seuil=1e-8; 
EnsPtContr = find(1-y.*f > seuil);

% on entre dans la boucle...
max_corA=inf;

while arret == 0 & max_corA >1e-4;
    
    mem_cout = [mem_cout sum((max(0,1-y.*f)).^2)];
    mem_beta = [mem_beta sum(abs(b))];
    
    
    
    % 
    % Process Margin and gradient of cost function 
    R=zeros(nbptsapp,1);
    R(EnsPtContr)= 1-y(EnsPtContr).*f(EnsPtContr);
    cor =  x(EnsPtContr,:)'*diag(y(EnsPtContr)) * R(EnsPtContr);

  % Look for max gradient   
    [max_cor,i] = max(abs(cor(Abar))); 
    % if a hinge step is done max_cor is not equal to max_corA
    if isempty(A)
        max_corA=max_cor;
    else
        [max_corA] = max(abs(cor(A)));
    end
    %
    % This is for debugging purpose
    % you can look at cor(A) in order to verify KKT condition
    %     cor(A)
    %     if max_cor >anc_max_cor&& TT
    %         TT=0;
    %         keyboard
    %     else
    %         anc_max_cor=max_cor
    %     end
    
    
    if Auto_Ajout_PTS == 1 
        A           = [A Abar(i)];
        Abar(i)     = [];
        b           = [b ; 0];     
    end
    signe       = sign(cor(A));
    
    % debugging
%     if sum( (abs(cor(A))-abs(cor(A(1)))<1e-3))~=length(A)
%         keyboard
%     end;
    
    cardA         = length(A); 
    cardAbar      = length(Abar);
    
    
    % Gradient Descent Direction 
    % Brute force : Do it like in LRAS
    %     xsup          = x(EnsPtContr,A) * diag(signe);
    %     InvXtX        = inv(xsup'*xsup + eye(cardA)*lambda );    
    %     alpha         = ( sum(sum(InvXtX)) )^(-0.5);
    %     W             = sum(alpha*InvXtX,2).* signe;
    
    
    % Do it like in Rosset's paper
    % W= - alpha*InvXtX*signe;
    %  alphaaux   = x(EnsPtContr,A)'*x(EnsPtContr,A)*W
    
    %  Do it in a clever way
    Gsup = (x(EnsPtContr,A)* diag(signe))'*(x(EnsPtContr,A) * diag(signe)) + eye(cardA)*lambda ;
    Rchol    = chol(Gsup); 
    bchol    = Rchol'\ones(size(Gsup,1),1);
    ychol    = Rchol\bchol;
    alpha    = (sum(ychol))^(-0.5);
    W        = alpha*ychol.* signe;
    
    
    eps=1e-10;
    alpha_bar     = x(EnsPtContr,Abar)'*x(EnsPtContr,A)*W;
    
    % Look for different events
    
    gamma_out_xsup_moins = (max_corA - cor(Abar))./(alpha - alpha_bar);
    gamma_out_xsup_plus  = (max_corA + cor(Abar))./(alpha + alpha_bar);
    
    
    
    gamma_cand_1          = [gamma_out_xsup_moins; gamma_out_xsup_plus];
    gamma_cand_1          = gamma_cand_1(find(gamma_cand_1>eps)); % uniquement les valeurs positives des gammas
    
    
    gamma_cand_3          = - b./W;
    ind_gamma_cand_3_pos  = find(gamma_cand_3>eps);
    gamma_cand_3          = gamma_cand_3(ind_gamma_cand_3_pos);
    
    % Hinge
    gamma_cand_2          = (1 - y.*(x(:,A)*b)) ./ ( y.*(x(:,A)*W) );
    ind_gamma_cand_2_pos  = find(gamma_cand_2>eps);
    gamma_cand_2          = gamma_cand_2(ind_gamma_cand_2_pos);
    
    
    
    [gamma,ind_gamma]     = min([gamma_cand_1' gamma_cand_2' gamma_cand_3']);
    
    if length(A)>1
        gamma_OLS            = max(cor(A(1:end-1))./b(1:end-1));
    else
        gamma_OLS            = 1e-9;
    end
    b_OLS                = b + gamma_OLS * W;
    
    if isempty(gamma)
        if verbose == 1
            fprintf('ATTENTION, Pas de bonne valeur pour gamma !\n');
        end
        gamma = eps;
    end
    mouvement = -1;
    if ind_gamma <= length(gamma_cand_1) % cas classique
        if verbose == 1
            fprintf('Cas classique : ajout d''une variable dans A :%d,   %2.3f\n', cardA,max_corA);
        end
        Auto_Ajout_PTS = 1;
        b = b + gamma * W;
        mouvement =1;
        
    elseif ind_gamma <= length(gamma_cand_1)+length(gamma_cand_2) % ajout d'un point dans EnsPtContr
        if verbose == 1
            fprintf('Cas Hinge : mouvement dans B! %d   %d   %2.3f \n',cardA,length(EnsPtContr),max_cor);
        end
        Auto_Ajout_PTS = 0;
        b = b + gamma * W;
        mouvement = 2;
        
        % Update the set of points over the margin
        New=find(abs(1-y.*(x(:,A)*b)) <= seuil);
        Ind=find(ismember(EnsPtContr,New)) ;
        if ~isempty(Ind)
            EnsPtContr(Ind)=[];
        else
            EnsPtContr=[EnsPtContr;New];
        end;
        
        
    else
        if verbose == 1
            fprintf('Cas d''elimination de variable dans A\n');
        end
        Auto_Ajout_PTS = 0; % élimination d'un point de l'ensemble actif
        compt_backward = compt_backward+1;
        b = b + gamma * W;
        
        ind_gamma      = ind_gamma - ( length(gamma_cand_1)+length(gamma_cand_2) );
        ind_gamma      = ind_gamma_cand_3_pos(ind_gamma);
        Abar           = [Abar A(ind_gamma)];
        A(ind_gamma)   = [];
        b(ind_gamma)   = [];
        mouvement = 3;
        
    end
    
    
    
    f     = x(:,A)*b;
    
    
    if length(A) > Limites.nbmaxSV | compt_backward >Limites.nbmaxMoveBack 
        % il faut tout arreter !
        fprintf('Bornes de securite depassees.\n');
        
        break;
        
    end
    
    if Auto_Ajout_PTS == 1  % ne pas prendre de solution sous-optimale
        for i=1:nbborne
            RETENIR = 0;    
            if strcmp(AllBorne{i}.type,'sumBeta' ) & length(AllBorne{i}.borne)>0
                if sum(abs(Beta)) > AllBorne{i}.borne(1)  
                    RETENIR = 1;
                end
                
            elseif strcmp(AllBorne{i}.type,'pcSV' ) & length(AllBorne{i}.borne)>0
                if length(A)/nbdim > AllBorne{i}.borne(1)
                    RETENIR = 1;
                end
                
            elseif strcmp(AllBorne{i}.type,'nbSV') & length(AllBorne{i}.borne)>0
                if length(A) >= AllBorne{i}.borne(1)
                    RETENIR = 1;            
                end
                
            elseif strcmp(AllBorne{i}.type,'trapscale')  & length(AllBorne{i}.borne)>0
                if (A(end) > AllBorne{i}.indTS & A(end) < size(x,2)) % le point appartient a une echelle interdite
                    compt_trap = length(find(A > AllBorne{i}.indTS & A < size(x,2))); % combien ?
                    
                    if compt_trap > AllBorne{i}.borne(1) 
                        RETENIR = 1;    
                    end
                end
                
            elseif strcmp(AllBorne{i}.type,'RVtrap') & length(AllBorne{i}.borne)>0
                if Compt_RVpluscor > AllBorne{i}.borne(1)
                    RETENIR = 1;   
                end
                
            elseif strcmp(AllBorne{i}.type,'countback') & length(AllBorne{i}.borne)>0
                if compt_backward > AllBorne{i}.borne(1) 
                    RETENIR = 1;    
                end    
                
            elseif strcmp(AllBorne{i}.type,'pcNRJ') & length(AllBorne{i}.borne)>0
                if R'*R < AllBorne{i}.borne(1) * (y'*y) % borne sur l'energie du residu
                    RETENIR = 1;    
                end        
            end
            
            
            if RETENIR == 1
                solution{nbexpcur}.indxsup = A;
                solution{nbexpcur}.Beta    = b';%Beta(A);
                solution{nbexpcur}.type    = AllBorne{i}.type;
                if strcmp(AllBorne{i}.type,'trapscale'),
                    solution{nbexpcur}.borne   = AllBorne{i}.borne(1)*0.01 + AllBorne{i}.indTS;% recomposition !
                else
                    solution{nbexpcur}.borne   = AllBorne{i}.borne(1);
                end
                
                if test_OLS == 1
                    solution_OLS{nbexpcur}.indxsup = A;
                    solution_OLS{nbexpcur}.Beta    = b_OLS';
                    solution_OLS{nbexpcur}.type    = AllBorne{i}.type;% '_OLS'];
                    if strcmp(AllBorne{i}.type,'trapscale'),
                        solution_OLS{nbexpcur}.borne   = AllBorne{i}.borne(1)*0.01 + AllBorne{i}.indTS; % recomposition !
                    else
                        solution_OLS{nbexpcur}.borne   = AllBorne{i}.borne(1);
                    end
                    
                end
                
                
                %fprintf('%4d/%4d : %15s (%f) # %d\n',nbexpcur,nbexptot,solution{nbexpcur}.type,solution{nbexpcur}.borne,length(A));
                %fprintf('\b\b\b\b\b\b\b\b\b%4d/%4d',nbexpcur,nbexptot);
                fprintf('\n%4d/%4d',nbexpcur,nbexptot);
                
                AllBorne{i}.borne(1)     = []; % elimination !
                nbexpcur                 = nbexpcur +1;     
                %            size(A)
            end
        end
    end
    if length(A) == nbvar 
        fprintf('Regression totale !\n');
        break;
    end
    
    
    
    
    iteration = iteration + 1;
    
    if nbexpcur-1==nbexptot, % le compteur est en avance !
        break;
    end
end
% % % 

nbexpcur    = nbexpcur-1;

if nbexpcur == 0,
    solution     = [];
    solution_OLS = [];
    return;
end


for i=1:nbexpcur
    
    indb = find(solution{i}.indxsup == size(x,2));
    if length(indb) ~= 0
        solution{i}.b = solution{i}.Beta(indb) / sqrt(nbptsapp); % "denormalisation"
        solution{i}.Beta(indb) = [];
        solution{i}.indxsup(indb) = [];    
    else
        solution{i}.b = 0;
    end
    
    if test_OLS == 1
        if length(indb) ~= 0
            solution_OLS{i}.b = solution_OLS{i}.Beta(indb) / sqrt(nbptsapp); % "denormalisation"
            solution_OLS{i}.Beta(indb) = [];
            solution_OLS{i}.indxsup(indb) = [];    
        else
            solution_OLS{i}.b = 0;
        end
    end
end



