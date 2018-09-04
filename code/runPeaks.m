
clear;
clc;
addpath('/Users/bowei/Desktop/PhD in Biostatistics/Courses/CS584/');
%% creat data
% Y: sample matrix n*n_f
% C: response matrix n*_c
[Y, C] = setupPeaks();
n = size(Y,1);
n_f = size(Y, 2);
n_c = size(C, 2);
[~, trueClass] = max(C, [], 2);

%% split data
% 20% validation data  80% training data
rng(3);
num = (1:n);
perm = num(randperm(n));

trainIndex = perm(1: (0.6*n));
trueTrainClass = trueClass(trainIndex);
CTrain = C(trainIndex, :);
YTrain = Y(trainIndex,:);

validationIndex = perm((0.6*n+1):(0.8*n));
trueValidClass = trueClass(validationIndex);
CValid = C(validationIndex,:);
YValid = Y(validationIndex,:);

testIndex = perm((0.8*n+1):n);
trueTestClass = trueClass(testIndex);
CTest = C(testIndex, :);
YTest = Y(testIndex,:);



%%
%%linear classification
rng(4)
YTrain_new=[YTrain,YTrain.^2];
YValid_new=[YValid,YValid.^2];
alpha=0.5;
n_f_new=size(YTrain_new,2);

W0 = randn((n_f_new + 1)*n_c, 1);
tic;
[WEstimate_SDNW, EntropyNW, NormNW, WWNW, IterNW]=NWCG(@(W) softMax(W, YTrain_new, CTrain,alpha), W0, struct('maxIter',300,'stop',1e-6,'tol',1e-3));
a=toc;


[PredNW, ErrorNW] = prediction(WEstimate_SDNW, YTrain_new, CTrain);
[PredNWV, ErrorNWV] = prediction(WEstimate_SDNW, YValid_new, CValid);
%[PredNWT, ErrorNWT] = prediction(WEstimate_SDNW, YTest_new, CTest);
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
%fprintf('Newton PCG Testing Error =%1.3f\t\n', ErrorNWT) 
disp(a);
disp(IterNW);

%%
%%extreme learning
rng(4)
YTrain_new=[YTrain,YTrain.^2];
YValid_new=[YValid,YValid.^2];


p=120;
alpha=0.05;
n_f_new=size(YTrain_new,2);
YTrain_new = extremeML(YTrain_new, p, 4, @relu);
YValid_new = extremeML(YValid_new, p, 4, @relu);
W0 = randn((p + 1)*n_c, 1);
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
%fprintf('Newton PCG Testing Error =%1.3f\t\n', ErrorNWT) 
disp(a);
disp(IterNW);
