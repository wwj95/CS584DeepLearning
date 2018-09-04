% Yikai's singleLayer with sigma = tanh function. 
% Inputs:
% K: nf x m
% b: scalar
% Y: n x nf

function[Z,JKt,Jbt,JYt,JK,Jb,JY] = singleLayer(K,b,Y)
% function[Z] = singleLayer(K,b,Y)

    % Define activation functions: 
    f      = @(x) tanh(x);
    f1d = @(x) 1-tanh(x).^2; % first order derivative
    vec = @(V) V(:);
    
    Z0 = Y*K+b;
    
    Z = f(Z0);
    
    JKt = @(w) Y'*(f1d(Z0).*reshape(w,size(Z0)));
    JK  = @(v) f1d(Z0).*(Y*reshape(v,size(K)));
    
    Jbt = @(w) sum(sum(f1d(Z0).*reshape(w,size(Z0))));
    Jb  = @(x) f1d(Z0)*x;
    
    JY  = @(v) f1d(Z0).*(reshape(v,size(Y))*K);
    JYt = @(v) (f1d(Z0).* reshape(v,size(Z0)))*K';
    
end
