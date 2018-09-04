% !gunzip *.gz
clear;

Ytest = loadMNISTImages('t10k-images-idx3-ubyte')';
ctest = loadMNISTLabels('t10k-labels-idx1-ubyte');
nval  = length(ctest);
Ctest = full(sparse(1:nval,ctest+1,ones(nval,1),nval,10));

Y = loadMNISTImages('train-images-idx3-ubyte')';
c = loadMNISTLabels('train-labels-idx1-ubyte');
nex = length(c);
C = full(sparse(1:nex,c+1,ones(nex,1),nex,10));


%% solve using QR (minimum norm solution)
tic;
W1 = Y\C;
toc;

%% solve using normal equations and cholesky
alpha = 1e-2;
tic;
W2 = (Y'*Y + alpha*speye(size(Y,2)))\(Y'*C);
toc;
%% visualize weights
idc = 9; % pick class
figure(1); clf;
subplot(1,2,1)
imagesc(reshape(W1(:,idc),28,28))
colorbar
subplot(1,2,2)
imagesc(reshape(W2(:,idc),28,28))
colorbar

%% gauge variance for different alphas
Cnoise = randn(size(C));
alphaSmall = 1e-2;
tic;
W2 = (Y'*Y + alphaSmall*speye(size(Y,2)))\(Y'*Cnoise);
toc;

alphaBig = 1e0;
tic;
W3 = (Y'*Y + alphaBig*speye(size(Y,2)))\(Y'*Cnoise);
toc;
