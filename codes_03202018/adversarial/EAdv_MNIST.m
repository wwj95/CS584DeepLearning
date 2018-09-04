% Example for adversarial attach on a single layer NN

close all; clear all; clc;

load ELM_MNISTweights.mat
addpath ../Data
Ytest = loadMNISTImages('t10k-images.idx3-ubyte')';
ctest = loadMNISTLabels('t10k-labels.idx1-ubyte');

Y = loadMNISTImages('train-images.idx3-ubyte')';
c = loadMNISTLabels('train-labels.idx1-ubyte');

nval  = length(ctest);
Ctest = full(sparse(1:nval,ctest+1,ones(nval,1),nval,10));
nex = length(c);
C = full(sparse(1:nex,c+1,ones(nex,1),nex,10));
%% test weights
Z = singleLayer(elmK,elmb,Ytest);
h = exp(Z*elmW)./sum(exp(Z*elmW),2);
[~,ind] = max(h,[],2);
Cv = zeros(size(Ctest));
Ind = sub2ind(size(Cv),[1:size(Cv,1)]',ind);
Cv(Ind) = 1;
valErr   = 100*nnz(abs(Cv-Ctest))/2/nnz(Cv);
fprintf('Validation Error %3.2f%%\n',valErr);

%% pick an example
id = 750;
y = Ytest(id,:);
cp = Cv(id,:);
c  = Ctest(id,:);
fig = figure(1); clf;
fig.Name = sprintf('Adversarial example: True label %d',find(cp)-1)
subplot(1,2,1)
imagesc(reshape(y,28,28))
title(sprintf('predicted label %d',find(c)-1))
set(gca,'FontSize',20);


