function [Ec,dE] = conn1DLayerNNObjFun(x,Y,C,m)
% [Ec,dE] = connLayerNNObjFun(x,Y,C,m)
%
% evaluates single layer and computes cross entropy, gradient and approx. Hessian
%
% Let x = [theta(:);b(:);W(:)], we compute
%
% E(x) = E(Z*W,C),   where Z = activation(Y*K(theta)+b)
%
% Inputs
%
%   x - current iterate, x=[theta(:);b(:);W(:)]
%   Y - input features
%   C - class probabilities
%   m - length(theta), used to split x correctly
%
% Output:
%
%   Ec - current value of loss function
%   dE - gradient w.r.t. theta,b,W, vector


    [~,nf] = size(Y);
    nc     = size(C,2);

    % split x into theta,b,W
    x = x(:);
    theta = x(1:m);
    b = x(m+1);
    W = reshape(x(m+2:end),[],nc);

    % evaluate conn layer
    [Z,Jthetat,Jbt] = conn1DLayer(theta,b,Y);

    % call cross entropy
    [Ec,dEW,d2EW,dEZ,d2EZ] = softMax(W,Z,C);

    if nargout>1
        dEtheta = Jthetat(dEZ); 
        dEb     = Jbt(dEZ);
        dE      = [dEtheta(:); dEb(:); dEW(:)];
    end

end





