% Yikai's singleLayer with sigma = tanh function. 
% Inputs:
% theta: m x 1
% b: scalar
% Y: n x nf

function[Z,Jthetat,Jbt] = conn2DLayer(theta,b,Y,sTheta,nImg)

    % Define activation functions: 
    f      = @(x) tanh(x);
    f1d = @(x) 1-tanh(x).^2; % first order derivative
    vec = @(V) V(:);
    

    [YKmv,~,~,Jtmv] = conv2D(nImg,sTheta,theta,Y);
    
    Z0 = YKmv(Y)+b;
    Z = f(Z0);
    
    
    Jthetat = @(w) Jtmv( f1d(Z0) .* reshape(w,size(Z0)) );
    
    JKt = @(w) Y'*(f1d(Z0).*reshape(w,size(Z0)));
    
    Jbt = @(w) sum(sum(f1d(Z0).*reshape(w,size(Z0))));
    
end
