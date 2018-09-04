function [YKmv,YKtmv,Jmv,Jtmv] = conv1D(nf,theta,Y)
% [YKmv,YKtmv,Jmv,Jtmv] = conv1D(nf,theta,Y)
%
% computes 1D convolutions with FFT
%
%  Z = Y * K(theta),
%
% where size(Y)=n x nf and size(K)= nf x nf.
%
% Input:
%   nf    - number of features
%   theta - stencil (assumed to be odd number of elements)
%   Y     - features (needed only for Jacobians)
%
% Output
%   YKmv  - function handle for Y -> Y*K(theta)
%   YKtmv - function handle for Y -> Y*K(theta)'
%   Jmv   - function handle for v -> J(Y*K(theta)) * v
%   Jtmv  - function handle for w -> J(Y*K(theta))'* w


if nargin==0
    testThisMethod
    return;
end
lam   = fft(getK1(theta,nf));
sdiag = @(v) spdiags(v(:),0,numel(v),numel(v));
% your code here
YKmv  = @(Y) real(fft(sdiag(lam)*ifft(Y')))'; 
YKtmv = @(Y) real(ifft(sdiag(lam)*fft(Y')))';

% Jacobians
if nargout>2
    iFy = ifft(Y');
    q   = getK1(1:numel(theta),nf);
    I   = find(q);
    J   = q(I);
    Q   = sparse(I,J,ones(numel(theta),1),nf,numel(theta));
    
    Jmv  = @(v) real(fft(sdiag(fft(Q*v))*iFy))'; 
    Jtmv = @(w) real(Q'*fft(sum(iFy.*fft(w'),2)));
end


% ---- helper functions ----

function K1 = getK1(theta,m)
% builds first column of K(theta)
center = (numel(theta)+1)/2;
K1 = circshift([theta(:);zeros(m-numel(theta),1)],1-center);



% ----- test function -----
function testThisMethod
nf    = 16;
n     = 10;
theta = randn(3,1);
K     = full(spdiags(ones(nf,1)*flipud(theta)',-1:1,nf,nf));
K(1,end) = theta(3);
K(end,1) = theta(1);
Y = randn(n,nf);

[YKmv,YKtmv,Jmv,Jtmv] = feval(mfilename,nf,theta,Y);

T1 = Y*K;
T2 = YKmv(Y);
fprintf('error for Y*K:   %1.2e\n',norm(T1-T2))

T1 = Y*K';
T2 = YKtmv(Y);
fprintf('error for Y*K'':  %1.2e\n',norm(T1-T2))

% derivative check
dth = randn(size(theta));
Ykmv = feval(mfilename,nf,theta+dth,Y);
T1 = Ykmv(Y);
T2 = YKmv(Y) + Jmv(dth);
fprintf('error for J*v:   %1.2e\n',norm(T1-T2));

% adjoint check
dZ = randn(size(Y));
T1 = sum(sum(dZ.*Jmv(dth)));
T2 = sum(sum(dth.*Jtmv(dZ)));
fprintf('adjoint error:   %1.2e\n',norm(T1-T2));

