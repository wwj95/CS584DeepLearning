close all; clear all; clc;

%% get peaks data
np = 8000;  % num of points sampled
nc = 5;     % num of classes
ns = 256;   % length of grid

[Y, C] = setupPeaks(np, nc, ns);

numTrain = size(Y, 1)*0.80;
idx = randperm(size(Y,1));
idxTrain = idx(1:numTrain);
idxValid = idx(numTrain+1:end);

YTrain = Y(idxTrain, :);
CTrain = C(idxTrain, :);

YValid = Y(idxValid, :);
CValid = C(idxValid, :);

[YTest, CTest] = setupPeaks(2000, nc, ns);

nf = size(Y,2);
nc = size(C,2);
%% optimize
% m = 640/20;
m = 96;
KOpt = randn(2,m);
bOpt = randn(1);
W0   = randn(m,5);

%% compare nonlinearities
figure(1); clf;
subplot(2,2,1);
Z1 = sin(Y*KOpt+bOpt);
Z2 = tanh(Y*KOpt+bOpt);
Z3 = max(0,Y*KOpt+bOpt);
semilogy(svd(Z1),'linewidth',3);
hold on;
semilogy(svd(Z2),'linewidth',3);
semilogy(svd(Z3),'linewidth',3);
legend('sin','tanh','relu');
title('singular values')
set(gca,'FontSize',20)
%% optimize
relu = @(x) max(x,0);
acts = {@sin,@tanh,relu};

for k=1:numel(acts)
    act  = acts{k};
    Z    = act(Y*KOpt+bOpt);
    fctn = @(x) softMax(x,Z,C);
    param = struct('maxIter',300,'maxStep',1,'tolCG',1e-1,'maxIterCG',100);
    WOpt = gaussNewtonPCG(fctn,W0(:),param);
    %%
    WOpt = reshape(WOpt,m,nc);
    Strain = Z*WOpt;
    S      = act(YTest*KOpt+bOpt)*WOpt;
    htrain = exp(Strain)./sum(exp(Strain),2);
    h      = exp(S)./sum(exp(S),2);
    
    % Find the largesr entry at each row
    [~,ind] = max(h,[],2);
    Cv = zeros(size(CTest));
    Ind = sub2ind(size(Cv),[1:size(Cv,1)]',ind);
    Cv(Ind) = 1;
    [~,ind] = max(htrain,[],2);
    Cpred = zeros(size(C));
    Ind = sub2ind(size(Cpred),[1:size(Cpred,1)]',ind);
    Cpred(Ind) = 1;
    
    trainErr = 100*nnz(abs(C-Cpred))/2/nnz(C);
    valErr   = 100*nnz(abs(Cv-CTest))/2/nnz(Cv);
    %%
    x = linspace(-3,3,201);
    [Xg,Yg] = ndgrid(x);
    Z = act([Xg(:) Yg(:)]*KOpt+bOpt)*WOpt;
    h      = exp(Z)./sum(exp(Z),2);
    
    [~,ind] = max(h,[],2);
    Cpred = zeros(numel(Xg),5);
    Ind = sub2ind(size(Cpred),[1:size(Cpred,1)]',ind);
    Cpred(Ind) = 1;
    img = reshape(Cpred*(1:5)',size(Xg));
    %%
    figure(1);
    subplot(2,2,1+k)
    imagesc(x,x,img')
    title(sprintf('%s - train %1.2f%% val %1.2f%%',func2str(act),trainErr,valErr));
    
end

