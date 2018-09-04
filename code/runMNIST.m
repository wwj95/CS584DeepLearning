
clear;
clc;
addpath('/Users/bowei/Desktop/PhD in Biostatistics/Courses/CS584/');
%% load data
[yTrain, cTrain, YTest, CTest] = loadMNIST();

%% split train into train and validation
rng(2);
n = size(yTrain, 1);
num = (1:n);
perm = num(randperm(n));

trainIndex = perm(1: (0.8*n));
CTrain = cTrain(trainIndex, :);
YTrain = yTrain(trainIndex,:);
[~, trueTrainClass] = max(CTrain, [], 2);

validationIndex = perm((0.8*n+1):n);
CValid = cTrain(validationIndex, :);
YValid = yTrain(validationIndex,:);
[~, trueValidClass] = max(CValid, [], 2);

[~, trueTestClass] = max(CTest, [], 2);
%% variable setup
n_f = size(YTrain, 2);
n_c = size(CTrain, 2);
n = size(YTrain, 1);


%%
%Linear Classification
rng(3)
YTrain_new=YTrain;
YValid_new=YValid;
YTest_new=YTest;
W0 = randn((n_f+ 1)*n_c, 1);
alpha=0.05;
tic;
[WEstimate_SDNW, EntropyNW, NormNW, WWNW, IterNW]=NWCG(@(W) softMax(W, YTrain_new, CTrain,alpha), W0, struct('maxIter',300,'stop',1e-6,'tol',1e-3));
a=toc;


[PredNW, ErrorNW] = prediction(WEstimate_SDNW, YTrain_new, CTrain);
[PredNWV, ErrorNWV] = prediction(WEstimate_SDNW, YValid_new, CValid);
[PredNWT, ErrorNWT] = prediction(WEstimate_SDNW, YTest_new, CTest);
[PredWNW, ErrorWNW] = prediction(WWNW, YTrain_new, CTrain);
[PredVNW, ErrorVNW] = prediction(WWNW, YValid_new, CValid);

figure(2)
subplot(1, 2, 1)
plot(1:size(ErrorWNW,2),ErrorWNW,'b',1:size(ErrorVNW,2),ErrorVNW,'r')
legend('Train','Validation')
title("Classification Error")
subplot(1, 2, 2)
plot(EntropyNW)
title("Entropy")

fprintf('Newton PCG Training Error =%1.3f\t\n', ErrorNW) 
fprintf('Newton PCG Validation Error =%1.3f\t\n', ErrorNWV) 
fprintf('Newton PCG Testing Error =%1.3f\t\n', ErrorNWT) 
disp(a);
disp(IterNW);
%%
%Extreme learning
rng(3);
p=1000;
alpha=0.05;
YTrain_new=extremeML(YTrain,p,3,@relu);
YValid_new=extremeML(YValid,p,3,@relu);
YTest_new=extremeML(YTest,p,3,@relu);
W0 = randn((p+ 1)*n_c, 1);

tic;
[WEstimate_SDNW, EntropyNW, NormNW, WWNW, IterNW]=NWCG(@(W) softMax(W, YTrain_new, CTrain,alpha), W0, struct('maxIter',300,'stop',1e-6,'tol',1e-3));
a=toc;


[PredNW, ErrorNW] = prediction(WEstimate_SDNW, YTrain_new, CTrain);
[PredNWV, ErrorNWV] = prediction(WEstimate_SDNW, YValid_new, CValid);
[PredNWT, ErrorNWT] = prediction(WEstimate_SDNW, YTest_new, CTest);
[PredWNW, ErrorWNW] = prediction(WWNW, YTrain_new, CTrain);
[PredVNW, ErrorVNW] = prediction(WWNW, YValid_new, CValid);

figure(2)
subplot(1, 2, 1)
plot(1:size(ErrorWNW,2),ErrorWNW,'b',1:size(ErrorVNW,2),ErrorVNW,'r')
legend('Train','Validation')
title("Classification Error")
subplot(1, 2, 2)
plot(EntropyNW)
title("Entropy")

fprintf('Newton PCG Training Error =%1.3f\t\n', ErrorNW) 
fprintf('Newton PCG Validation Error =%1.3f\t\n', ErrorNWV) 
fprintf('Newton PCG Testing Error =%1.3f\t\n', ErrorNWT) 
disp(a);
disp(IterNW);
