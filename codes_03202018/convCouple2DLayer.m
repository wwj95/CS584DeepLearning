function[Z,Jthetat,Jbt] = convCouple2DLayer(theta,b,Y,sTheta,nImg)

    % Define activation functions: 
    f      = @(x) tanh(x);
    f1d = @(x) 1-tanh(x).^2; % first order derivative
    vec = @(V) V(:);
    

    [YKmv,~,~,Jtmv] = convCoupled2D(nImg,sTheta,theta,Y);
    
    Z0 = YKmv(Y)+b;
    Z = f(Z0);
    
    
    Jthetat = @(w) Jtmv( f1d(Z0) .* reshape(w,size(Z0)) );
    
    JKt = @(w) Y'*(f1d(Z0).*reshape(w,size(Z0)));
    
    Jbt = @(w) sum(sum(f1d(Z0).*reshape(w,size(Z0))));
    
end
