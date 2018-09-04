function [YKmv,YKtmv,Jmv,Jtmv] = convCoupled2D(nImg,sTheta,theta,Y)
% function [YKmv,YKtmv,Jmv,Jtmv] = convCoupled2D(nImg,sTheta,theta,Y)
%
% computes the coupled convolution of multi-channel images Y
%
%
% Input:
%  nImg   -  number of pixels, e.g., nImg = [16,16];
%  sTheta -  size of kernel, e.g., sTheta = [3,3,4,6] for 3x3 convolutions
%            applied to 4 input channels giving 6 output channels
%  theta  -  weights
%  Y      -  feature matrix, only needed for derivative computation
%
% Output: 
%  YKmv   -  function handle for computing Y -> Y*K(theta)
%  YKtmv  -  function handle for computing Y -> Y*K(theta)'
%  Jmv    -  function handle for computing v -> Y*K(theta+v)
%  Jtmv   -  function handle for computing Z -> Jac'*Z


if nargin==0
    nImg   = [16 16];
    sTheta = [3 3 4 6];
    n      = 1;
    nf     = prod(nImg)*sTheta(3);
    theta  = ones(prod(sTheta),1);
    Y      = randn(n,nf);
    
    % uncomment to test your GPU!
%     Y = gpuArray(single(Y));
%     theta = gpuArray(single(theta));
    
    [YKmv,YKtmv,Jmv,Jtmv] = feval(mfilename,nImg,sTheta,theta,Y);
    YK = YKmv(Y);
    Z  = randn(size(YK),'like',YK);
    YKtZ = YKtmv(Z);
    t1 = sum(vec(YK.*Z));
    t2 = sum(vec(YKtZ.*Y));
    fprintf('adjoint  error:     %1.2e\n',norm(t1-t2))
    
    
    % derivative check
    dth = randn(size(theta),'like',theta);
    Kmv2 = feval(mfilename,nImg,sTheta,theta+dth,Y);
    T1 = Kmv2(Y);
    T2 = YKmv(Y) + Jmv(dth);
    fprintf('error in Jacobian: %1.2e\n',norm(T1-T2))
    
    % adjoint check
    dZ = randn(size(T2),'like',theta);
    dth = randn(size(dth),'like',theta);
    T1 = sum(sum(dZ.*Jmv(dth)));
    T2 = sum(sum(dth(:)'*Jtmv(dZ)));
    fprintf('adjoint Jacobian error:     %1.2e\n',norm(T1-T2))
    return
end

YKmv  = @(Y) Amv(nImg,sTheta,theta,Y); 
YKtmv = @(Y) Atmv(nImg,sTheta,theta,Y);

if nargout>2
    Jmv  = @(v) Amv(nImg,sTheta,v,Y); 
    Jtmv = @(Z) JthetaTmv(nImg,sTheta,Y,Z); 
end

function x=vec(x)
x=x(:);

function Z = Amv(nImg,sTheta,theta,Y)
% compute convolution

nex   = numel(Y)/(prod(nImg)*sTheta(3));

Z    = zeros([nImg sTheta(4) nex],'like',Y); 
S     = reshape(fft2(getK1(theta,nImg,sTheta)),[nImg sTheta(3:4)]);
Yh    = ifft2(reshape(Y',[nImg sTheta(3) nex]));
for k=1:sTheta(4)
    T  = S(:,:,:,k) .* Yh;
    Z(:,:,k,:)  = sum(T,3);
end
Z = real(fft2(Z));
Z  = reshape(Z,[],nex)';

function Y = Atmv(nImg,sTheta,theta,Z)
% compute transpose of convolution

nex =  numel(Z)/(prod(nImg)*sTheta(4));
Y = zeros([nImg sTheta(3) nex],'like',Z);
S   = reshape(fft2(getK1(theta,nImg,sTheta)),[nImg sTheta(3:4)]);

Zh = fft2(reshape(Z',[nImg sTheta(4) nex]));
for k=1:sTheta(3)
    Sk = squeeze(S(:,:,k,:));
    Y(:,:,k,:) = sum(Sk.*Zh,3);
end
Y = real(ifft2(Y));
Y = reshape(Y,[],nex)';
            
function dtheta = JthetaTmv(nImg,sTheta,Y,Z)
% compute Jac'*Z

nex    =  numel(Y)/(prod(nImg)*sTheta(3));

dth1 = zeros(prod(sTheta(1:3)),sTheta(4),'like',Y);
Yh   = permute(ifft2(reshape(Y',[nImg sTheta(3) nex])),[1 2 4 3]);
Zh   = fft2(reshape(Z',[nImg sTheta(4) nex]));

% get q vector for a given row in the block matrix
v   = vec(1:prod(sTheta(1:3)));
q   = getK1(v,nImg,sTheta);

I    = find(q(:));
for k=1:sTheta(4)
    Zk = squeeze(Zh(:,:,k,:));
    tt = squeeze(sum(Zk.*Yh,3));
    tt = real(fft2(tt));
    dth1(q(I),k) = tt(I);
end
dtheta = dth1(:);

function K1 = getK1(theta,nImg,sTheta)
% compute first row of convolution matrix
theta = reshape(theta,sTheta(1),sTheta(2),[]);
center = (sTheta(1:2)+1)/2;

K1  = zeros([nImg size(theta,3)],'like',theta);
K1(1:sTheta(1),1:sTheta(2),:) = theta;
K1  = circshift(K1,1-center);

