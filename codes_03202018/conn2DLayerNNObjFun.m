function [Ec,dE] = conn2DLayerNNObjFun(x,Y,C,m,sTheta,nImg)

    [~,nf] = size(Y);
    nc     = size(C,2);

    % split x into theta,b,W
    x = x(:);
    theta = x(1:m);
    b = x(m+1);
    W = reshape(x(m+2:end),[],nc);

    % evaluate conn layer
    [Z,Jthetat,Jbt] = conn2DLayer(theta,b,Y,sTheta,nImg);

    % call cross entropy
    [Ec,dEW,d2EW,dEZ,d2EZ] = softMax(W,Z,C);

    if nargout>1
        dEtheta = Jthetat(dEZ); 
        dEb     = Jbt(dEZ);
        dE      = [dEtheta(:); dEb(:); dEW(:)];
    end

end





