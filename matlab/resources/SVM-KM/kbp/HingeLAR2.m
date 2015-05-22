function [solution, solution_OLS] = HingeLAR(x,y, AllBorne, Limites, lambda, verbose, xappforplot)

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
% init  : b = 0;
nbvar=size(x,2)
f = zeros(nbptsapp,1);
A = [];
Abar = 1:nbvar; 
beta = zeros(nbvar,1);


Auto_Ajout_PTS = 1;
iteration = 0;
%
mem_cor = [];
mem_beta = [];
mem_cout = [];
arret = 0;



R=max(0,1-y.*(x*beta));
cor =  2*x'*diag(y) * R;
[max_cor,i] = max(abs(cor(Abar)));
A=[A Abar(i)];
Abar(i)=[];

gamma = zeros(nbvar,1);
gamma(A)=-sign(cor(A));

while arret == 0
    

    
    corA=cor(A(1)); % un seul suffit
    corAbar=cor(Abar);
    alpha     = x'*x*gamma;
    alphaA=alpha(A(1));
    alphaAbar=alpha(Abar);
    
    
    gamma_out_xsup_moins = (corA- corAbar)./(alphaA - alphaAbar);
    gamma_out_xsup_plus  = (corA + corAbar)./(alphaA + alphaAbar);
    gamma_cand_1          = [gamma_out_xsup_moins; gamma_out_xsup_plus];
    ind_gamma_cand_1      =find(gamma_cand_1>eps);
    aux                     =[Abar Abar];
    Liste1                  =aux(ind_gamma_cand_1) ; 
    gamma_cand_1          = gamma_cand_1(ind_gamma_cand_1 ); 
    
    gamma_cand_2          = (1 - y.*(x*beta)) ./ ( y.*(x*gamma) );
    Liste2 = find(gamma_cand_2>eps)';
    gamma_cand_2          = gamma_cand_2(Liste2);
    
    gamma_cand_3          = - beta(A)./alpha(A); % .* signe(indpb)'ind_gamma_cand_1  ;
    Liste3 = find(gamma_cand_3>eps);
    gamma_cand_3          = gamma_cand_3(Liste3);
    
    [Step,ind_gamma]     = min([gamma_cand_1'  gamma_cand_3' gamma_cand_2']);
    ListeVar=[Liste1 Liste3 Liste2];
    
    
    
            cardA         = length(A); 
    if ind_gamma <= length(gamma_cand_1) % cas classique
        if verbose == 1
            fprintf('Cas classique : ajout d''une variable dans A :%d\n', cardA);
        end
        beta = beta + Step*gamma;
        
        A=[A ListeVar(ind_gamma)];
        Abar(ListeVar(ind_gamma))=[];
        mouvement =1;
        
    elseif ind_gamma <= length(gamma_cand_1)+length(gamma_cand_3) % ajout d'un point dans EnsPtContr
        if verbose == 1
            fprintf('Cas d''elimination de variable dans A\n');
        end
        
        compt_backward = compt_backward+1;
        beta = beta + Step * gamma;
        
        Abar           = [Abar ListeVar(ind_gamma)];
        A(ListeVar(ind_gamma))  = [];
        beta(ListeVar(ind_gamma))   =0;
        mouvement = 3;
        
    else
        if verbose == 1
    %        fprintf('Cas Hinge : mouvement dans B! %d   %d\n',cardA,length(EnsPtContr));
        end
        
        beta = beta + Step*gamma;
        
        
    end
    
    % Update 
        cardA         = length(A); 
    cardAbar      = length(Abar);
    EnsPtContr = find(1-y.*(x*beta) >= 0);
    xsup          = x(EnsPtContr,A);
    InvXtX        = inv(2*xsup'*xsup + eye(cardA)*lambda );
    gamma = zeros(nbvar,1);
    gamma(A)     = - InvXtX*sign(beta(A));
    
    R=max(0,1-y.*(x*beta));
    cor =  2*x'*diag(y) * R;
    [max_cor,i] = max(abs(cor(Abar)));
    max_cor
end;


%     
%     
%     
%     % on entre dans la boucle...
%     while arret == 0,
%         
%         mem_cout = [mem_cout sum((max(0,1-y.*f)).^2)];
%         mem_beta = [mem_beta sum(abs(b))];
%         
%         %%%napp = 1;% length(x);
%         EnsPtContr = find(1-y.*f >= 0);
%         
%         R    = max(0,1-y.*f);
%         
%         cor =  x(EnsPtContr,:)'*diag(y(EnsPtContr)) * R(EnsPtContr);
%         
%         [max_cor,i] = max(abs(cor(Abar)));
%         
%         
%         if Auto_Ajout_PTS == 1 
%             A           = [A Abar(i)];
%             Abar(i)     = [];
%             b           = [b ; 0];     
%         end
%         signe       = sign(cor(A));
%         
%         
%         %    *diag(signe)
%         cardA         = length(A); 
%         cardAbar      = length(Abar);
%         
%         %     xsup          = x(EnsPtContr,A) * diag(signe);
%         %     InvXtX        = inv(xsup'*xsup + eye(cardA)*lambda );    
%         % alpha         = ( sum(sum(InvXtX)) )^(-0.5);
%         %     W             = sum(alpha*InvXtX,2).* signe;
%         
%         
%         Gsup = (x(EnsPtContr,A)* diag(signe))'*(x(EnsPtContr,A) * diag(signe)) + eye(cardA)*lambda ;
%         Rchol    = chol(Gsup); 
%         bchol    = Rchol'\ones(size(Gsup,1),1);
%         ychol    = Rchol\bchol;
%         alpha    = (sum(ychol))^(-0.5);
%         W        = alpha*ychol.* signe;
%         
%         
%         eps=1e-10;
%         %alpha_bar     = x'*x(:,A)*W; %x(:,Abar)'*x(:,A)*W;
%         alpha_bar     = x(:,Abar)'*x(:,A)*W;
%         %alpha_bar     = alpha_bar(Abar);
%         
%         gamma_out_xsup_moins = (max_cor - cor(Abar))./(alpha - alpha_bar);
%         gamma_out_xsup_plus  = (max_cor + cor(Abar))./(alpha + alpha_bar);
%         
%         gamma_cand_1          = [gamma_out_xsup_moins; gamma_out_xsup_plus];
%         gamma_cand_1          = gamma_cand_1(find(gamma_cand_1>eps)); % uniquement les valeurs positives des gammas
%         
%         gamma_cand_2          = (1 - y.*(x(:,A)*b)) ./ ( y.*(x(:,A)*W) );
%         ind_gamma_cand_2_pos  = find(gamma_cand_2>eps);
%         gamma_cand_2          = gamma_cand_2(ind_gamma_cand_2_pos);
%         
%         gamma_cand_3          = - b./W; % .* signe(indpb)');
%         ind_gamma_cand_3_pos  = find(gamma_cand_3>eps);
%         gamma_cand_3          = gamma_cand_3(ind_gamma_cand_3_pos);
%         
%         [gamma,ind_gamma]     = min([gamma_cand_1' gamma_cand_2' gamma_cand_3']);
%         if gamma < 1e-6
%             keyboard
%         end;
%         if length(A)>1
%             gamma_OLS            = max(cor(A(1:end-1))./b(1:end-1));
%         else
%             gamma_OLS            = 1e-9;
%         end
%         b_OLS                = b + gamma_OLS * W;
%         
%         if isempty(gamma)
%             if verbose == 1
%                 fprintf('ATTENTION, Pas de bonne valeur pour gamma !\n');
%             end
%             gamma = eps;
%         end
%         mouvement = -1;
%         if ind_gamma <= length(gamma_cand_1) % cas classique
%             if verbose == 1
%                 fprintf('Cas classique : ajout d''une variable dans A :%d\n', cardA);
%             end
%             Auto_Ajout_PTS = 1;
%             b = b + gamma * W;
%             mouvement =1;
%             
%         elseif ind_gamma <= length(gamma_cand_1)+length(gamma_cand_2) % ajout d'un point dans EnsPtContr
%             if verbose == 1
%                 fprintf('Cas Hinge : mouvement dans B! %d   %d\n',cardA,length(EnsPtContr));
%             end
%             Auto_Ajout_PTS = 0;
%             b = b + gamma * W;
%             mouvement = 2;
%             
%         else
%             if verbose == 1
%                 fprintf('Cas d''elimination de variable dans A\n');
%             end
%             Auto_Ajout_PTS = 0; % élimination d'un point de l'ensemble actif
%             compt_backward = compt_backward+1;
%             b = b + gamma * W;
%             
%             ind_gamma      = ind_gamma - ( length(gamma_cand_1)+length(gamma_cand_2) );
%             ind_gamma      = ind_gamma_cand_3_pos(ind_gamma);
%             Abar           = [Abar A(ind_gamma)];
%             A(ind_gamma)   = [];
%             b(ind_gamma)   = [];
%             mouvement = 3;
%             
%         end
%         
%         
%         
%         f     = x(:,A)*b;
%         %     ft    = xt(:,A)*b;
%         
%         %     if mouvement ~=2
%         %         plot2Ddec(xforplot, y, A, xgrid_1, xgrid_2, ft,1 ,1);
%         %         drawnow;
%         %         pause(0.3)
%         %     end
%         
%         
%         if length(A) > Limites.nbmaxSV | compt_backward >Limites.nbmaxMoveBack 
%             % il faut tout arreter !
%             fprintf('Bornes de securite depassees.\n');
%             
%             break;
%             
%         end
%         
%         if Auto_Ajout_PTS == 1  % ne pas prendre de solution sous-optimale
%             for i=1:nbborne
%                 RETENIR = 0;    
%                 if strcmp(AllBorne{i}.type,'sumBeta' ) & length(AllBorne{i}.borne)>0
%                     if sum(abs(Beta)) > AllBorne{i}.borne(1)  
%                         RETENIR = 1;
%                     end
%                     
%                 elseif strcmp(AllBorne{i}.type,'pcSV' ) & length(AllBorne{i}.borne)>0
%                     if length(A)/nbdim > AllBorne{i}.borne(1)
%                         RETENIR = 1;
%                     end
%                     
%                 elseif strcmp(AllBorne{i}.type,'nbSV') & length(AllBorne{i}.borne)>0
%                     if length(A) >= AllBorne{i}.borne(1)
%                         RETENIR = 1;            
%                     end
%                     
%                 elseif strcmp(AllBorne{i}.type,'trapscale')  & length(AllBorne{i}.borne)>0
%                     if (A(end) > AllBorne{i}.indTS & A(end) < size(x,2)) % le point appartient a une echelle interdite
%                         compt_trap = length(find(A > AllBorne{i}.indTS & A < size(x,2))); % combien ?
%                         
%                         if compt_trap > AllBorne{i}.borne(1) 
%                             RETENIR = 1;    
%                         end
%                     end
%                     
%                 elseif strcmp(AllBorne{i}.type,'RVtrap') & length(AllBorne{i}.borne)>0
%                     if Compt_RVpluscor > AllBorne{i}.borne(1)
%                         RETENIR = 1;   
%                     end
%                     
%                 elseif strcmp(AllBorne{i}.type,'countback') & length(AllBorne{i}.borne)>0
%                     if compt_backward > AllBorne{i}.borne(1) 
%                         RETENIR = 1;    
%                     end    
%                     
%                 elseif strcmp(AllBorne{i}.type,'pcNRJ') & length(AllBorne{i}.borne)>0
%                     if R'*R < AllBorne{i}.borne(1) * (y'*y) % borne sur l'energie du residu
%                         RETENIR = 1;    
%                     end        
%                 end
%                 
%                 
%                 if RETENIR == 1
%                     solution{nbexpcur}.indxsup = A;
%                     solution{nbexpcur}.Beta    = b';%Beta(A);
%                     solution{nbexpcur}.type    = AllBorne{i}.type;
%                     if strcmp(AllBorne{i}.type,'trapscale'),
%                         solution{nbexpcur}.borne   = AllBorne{i}.borne(1)*0.01 + AllBorne{i}.indTS;% recomposition !
%                     else
%                         solution{nbexpcur}.borne   = AllBorne{i}.borne(1);
%                     end
%                     
%                     if test_OLS == 1
%                         solution_OLS{nbexpcur}.indxsup = A;
%                         solution_OLS{nbexpcur}.Beta    = b_OLS';
%                         solution_OLS{nbexpcur}.type    = AllBorne{i}.type;% '_OLS'];
%                         if strcmp(AllBorne{i}.type,'trapscale'),
%                             solution_OLS{nbexpcur}.borne   = AllBorne{i}.borne(1)*0.01 + AllBorne{i}.indTS; % recomposition !
%                         else
%                             solution_OLS{nbexpcur}.borne   = AllBorne{i}.borne(1);
%                         end
%                         
%                     end
%                     
%                     
%                     %fprintf('%4d/%4d : %15s (%f) # %d\n',nbexpcur,nbexptot,solution{nbexpcur}.type,solution{nbexpcur}.borne,length(A));
%                     %fprintf('\b\b\b\b\b\b\b\b\b%4d/%4d',nbexpcur,nbexptot);
%                     fprintf('\n%4d/%4d',nbexpcur,nbexptot);
%                     
%                     AllBorne{i}.borne(1)     = []; % elimination !
%                     nbexpcur                 = nbexpcur +1;     
%                     %            size(A)
%                 end
%             end
%         end
%         if length(A) == nbvar 
%             fprintf('Regression totale !\n');
%             break;
%         end
%         
%         
%         
%         
%         iteration = iteration + 1;
%         
%         if nbexpcur-1==nbexptot, % le compteur est en avance !
%             break;
%         end
%     end
%     % % % 
%     
%     nbexpcur    = nbexpcur-1;
%     
%     if nbexpcur == 0,
%         solution     = [];
%         solution_OLS = [];
%         return;
%     end
%     
%     
%     for i=1:nbexpcur
%         
%         indb = find(solution{i}.indxsup == size(x,2));
%         if length(indb) ~= 0
%             solution{i}.b = solution{i}.Beta(indb) / sqrt(nbptsapp); % "denormalisation"
%             solution{i}.Beta(indb) = [];
%             solution{i}.indxsup(indb) = [];    
%         else
%             solution{i}.b = 0;
%         end
%         
%         if test_OLS == 1
%             if length(indb) ~= 0
%                 solution_OLS{i}.b = solution_OLS{i}.Beta(indb) / sqrt(nbptsapp); % "denormalisation"
%                 solution_OLS{i}.Beta(indb) = [];
%                 solution_OLS{i}.indxsup(indb) = [];    
%             else
%                 solution_OLS{i}.b = 0;
%             end
%         end
%     end
%     


