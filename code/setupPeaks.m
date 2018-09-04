function[Y,C] = setupPeaks(np,nc)
% generates PEAKs example
if not(exist('np','var')) || isempty(np)
    np = 8000; % sample size
end

if not(exist('nc','var')) || isempty(nc)
    nc = 5; % number of class
end
ns = 256;
[xx,yy,cc] = peaks(ns);  %peaks is a funcion which used to generate matrix: xx and yy are coordinates matrix used for plots. 
t1 = linspace(min(xx(:)),max(xx(:)),ns); 
t2 = linspace(min(yy(:)),max(yy(:)),ns);

% Binarize it
mxcc = max(cc(:)); mncc = min(cc(:)); 
hc = (mxcc - mncc)/(nc);
ccb = zeros(size(cc));
for i=1:nc
    ii =  (mncc + (i-1)*hc)< cc & cc <= (mncc+i*hc);
    ccb(ii) = i-1;
end

figure(1); clf;
imagesc(t1,t2,reshape(ccb,ns,ns))
rng('default');
rng(2)

% draw same number of points per class
Y = [];
npc = ceil(np/nc);
for k=0:nc-1
   xk = [xx(ccb==k) yy(ccb==k)];
   inds = randi(size(xk,1),npc,1);
   
   Y = [Y; xk(inds,:)];
end

C = kron(eye(nc),ones(npc,1)); 
