function [W, Entropy, Normd2S, WW, Iter] = NWCG(E,W,param)
% Inputs:
%     E  - function that provides value and gradient
%     W  - starting guess
%  param - struct with parameters


maxIter   = param.maxIter; % max number of iterations
tol=param.tol; % tolerance of pcg
eps = param.stop; % stop criterion

WW1 = zeros(size(W,1),maxIter+1);
Entropy1 = zeros(maxIter, 1);
Normd2S = zeros(maxIter, 1);


for i=1:maxIter
    %  your code here
[~,J,dS,d2S]=E(W);
D=pcg(d2S,-dS,tol);
[~,J_new]=E(W+D);
    if J_new<=J
       D=D;
       if norm(D)<eps
                break;
       end
    else
        while J_new>J
         D=0.5*D;
         [~,J_new]=E(W+D);
            if norm(D)<eps
                break;
            end
        end
        if norm(D)<eps
                break;
        end
    end
        Normd2S(i)=norm(D);
        W=W+D;
        WW1(:,i+1)=W;
        entropy = E(W);
        Entropy1(i) = entropy;
        fprintf('Iter=%.0f\t\n', i);
        fprintf('Entropy=%.10f\t\n', entropy);
end
Iter=i;
WW=WW1(:,1:Iter);
Entropy=Entropy1(1:Iter);



end


   

