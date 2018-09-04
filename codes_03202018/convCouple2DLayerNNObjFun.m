function [Ec,dE] = convCouple2DLayerNNObjFun(x,Y,C,m,sTheta,nImg)

    [~,nf] = size(Y);
    nc     = size(C,2);

    % split x into theta,b,W
    x = x(:);
    theta = x(1:m);
    b = x(m+1);
    W = reshape(x(m+2:end),[],nc);

    
    if nargout == 1
        
        Z = convCouple2DLayer(theta,b,Y,sTheta,nImg);
        Ec = softMax(W,Z,C);
        
    end
    
    if nargout>1
        
        [Z,Jthetat,Jbt] = convCouple2DLayer(theta,b,Y,sTheta,nImg);
        [Ec,dEW,~,dEZ,~] = softMax(W,Z,C);
        dEtheta = Jthetat(dEZ); 
        dEb     = Jbt(dEZ);
        dE      = [dEtheta(:); dEb(:); dEW(:)];
        
    end

end





