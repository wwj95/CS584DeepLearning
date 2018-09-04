close all; clear all; clc;

rng(20)
n  = 500; nf = 50; nc = 10; m  = 40;
Wtrue = randn(m,nc);
Ktrue = randn(nf,m);
btrue = .1;

Y     = randn(n,nf);
Cobs  = exp(singleLayer(Ktrue,btrue,Y)*Wtrue);
Cobs  = Cobs./sum(Cobs,2);

%% test output sizes
x0 = randn(numel(Ktrue)+numel(btrue)+numel(Wtrue),1);

[Ec,dE] = singleLayerNNObjFun(x0,Y,Cobs,m);

if not(isscalar(Ec))
    error('objective function should return scalar'); 
end

if any(size(dE)~=size(x0));
    error('gradient should be a column vector');
end

%% check derivative
dx = randn(size(x0));

[Ec,dE] = singleLayerNNObjFun(x0,Y,Cobs,m);
dEdx = dE'*dx;

% dF = dF + 1e-2*randn(size(dF));
err    = zeros(30,3);
for k=1:size(err,1)
    h = 2^(-k);
    Et = singleLayerNNObjFun(x0+h*dx,Y,Cobs,m);
    
    err(k,:) = [h, norm(Ec-Et), norm(Ec+h*dEdx-Et)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
legend('E0','E1');

%% run steepest descent
F = @(x) singleLayerNNObjFun(x,Y,Cobs,m);
param = struct('maxIter',1000,'maxStep',1);
xSol = steepestDescent(F,x0,param);

Ksol = reshape(xSol(1:nf*m),nf,m);
bsol = xSol(nf*m+1);
Wsol = reshape(xSol(nf*m+2:end),m,nc);
%%
norm(Ktrue-Ksol)/norm(Ktrue)
norm(bsol-btrue)/norm(btrue)
norm(Wtrue-Wsol)/norm(Wtrue)

Cpred  = exp(singleLayer(Ksol,bsol,Y)*Wsol);
Cpred  = Cpred./sum(Cpred,2);
norm(Cobs-Cpred)/norm(Cobs)
