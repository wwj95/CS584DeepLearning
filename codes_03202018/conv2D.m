function [YKmv,YKtmv,Jmv,Jtmv] = conv2D(nImg,sTheta,theta,Y)

if nargin==0
    nImg  = [16 16];
    xa    = linspace(0,1,nImg(1));
    ya    = linspace(0,1,nImg(2));
    [X,Y] = ndgrid(xa,ya);
    n     =  1;
    theta = [-1 0 1; -1 0 1; -1 0 1];
%     theta = randn(size(theta))
    y     = X(:)+Y(:);
    
    [YKmv,YKtmv,Jmv,Jtmv] = feval(mfilename,nImg,size(theta),theta,y);
    
    T2 = YKmv(y);
    figure(1); clf;
    subplot(1,3,1);
    imagesc(reshape(y,nImg));
    subplot(1,3,2);
    imagesc(reshape(T2,nImg));
    
    
    T2 = YKtmv(y);
    subplot(1,3,3);
    imagesc(reshape(T2,nImg));
    
    % derivative check
    dth = randn(size(theta));
    Kmv2 = feval(mfilename,nImg,size(theta),theta+dth,y);
    T1 = Kmv2(y);
    T2 = YKmv(y) + Jmv(dth);
    fprintf('error in Jacobian: %1.2e\n',norm(T1-T2))
    
    % adjoint check
    dZ = randn(size(y));
    dth = randn(size(dth));
    T1 = sum(sum(dZ.*Jmv(dth)'));
    T2 = sum(sum(dth(:)'*Jtmv(dZ)));
    fprintf('adjoint error:     %1.2e\n',norm(T1-T2))
    return
end

% Input:
% Y : n x nf
% theta : p x 1
% nImg : [h l], h*l = nf
% sTheta = size of original Theta

rshp3D = @(Y) reshape(Y',nImg(1),nImg(2),[]);
rshp2D = @(Y) reshape(Y,prod(nImg),[])';
vec    = @(V) V(:);

lam   = fft2(getK1(theta,nImg,sTheta));
% your code here
YKmv  = @(Y) rshp2D(real(fft2(lam.*ifft2(rshp3D(Y))))); 
YKtmv = @(Y) rshp2D(real(ifft2(lam.*fft2(rshp3D(Y)))));

if nargout>2
    iFy = ifft2(rshp3D(Y));
    q   = getK1(1:numel(theta),nImg,sTheta);
    I   = find(q);
    J   = q(I);
    Q   = sparse(I,J,ones(numel(theta),1),prod(nImg),numel(theta));
    
    Jmv  = @(v) rshp2D(real(fft2(fft2(reshape(Q*v(:),nImg)).*iFy))); 
    Jtmv = @(w) real(Q'*vec(fft2(sum(iFy.*fft2(rshp3D(w)),3)))); 
end

function K1 = getK1(theta,nImg,sTheta)
theta = reshape(theta,sTheta);
K1 = zeros(nImg);
K1(1:sTheta(1), 1:sTheta(2)) = theta;
center = (sTheta+1)/2;
K1  = circshift(K1,1-center);


