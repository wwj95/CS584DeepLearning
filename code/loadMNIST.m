function [yTrain, cTrain, yTest, cTest] = loadMNIST()  
  clear; % remove all variables
  clc; % clear the command window

  % function to vectorize  
  vec = @(x) x(:);

  % read train and test data
  imagesTrain = loadMNISTImages('train-images.idx3-ubyte');
  labelsTrain = loadMNISTLabels('train-labels.idx1-ubyte');
  imagesTest = loadMNISTImages('t10k-images.idx3-ubyte');
  labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');

  % preprocess training and test data
  yTrain = imagesTrain';
  nex = length(labelsTrain);
  cTrain = zeros(nex,10);
  % full: convert sparse matrix to full matrix
  % sparse: create  a sparse matrix or convert full to sparse
  % c + 1: original label is from 0 to 9
  cTrain = full(sparse(1:nex,labelsTrain+1,ones(nex,1),nex,10));

  yTest = imagesTest';
  nex = length(labelsTest);
  cTest = zeros(nex,10);
  cTest = full(sparse(1:nex,labelsTest+1,ones(nex,1),nex,10));
end