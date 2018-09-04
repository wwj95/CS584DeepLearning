
%%%%%%%%%%%%%%%%%
% MNIST Dataset %
%%%%%%%%%%%%%%%%%
clear;
addpath('/Users/johnz/Desktop/Coursework/MATH Numerical Methods for DL/codes_03202018/')


Y = [];
C = [];
for i = 1:5
    
    dat = load(['/Users/johnz/Desktop/Coursework/MATH Numerical Methods for DL/Project/hw1/cifar-10-batches-mat/data_batch_' num2str(i) '.mat']);
    
    Y = [Y; dat.data];
    C = [C; dat.labels];

end



C = [C==0,C==1,C==2,C==3,C==4,C==5,C==6,C==7,C==8,C==9];

n  = size(Y,1);
nc = size(C,2);
nImg = [32 32]; % size of figure
nf = size(Y,2);

Y = double(Y);
Y = (Y - mean(reshape(Y,[nf*n 1])))/std(reshape(Y,[nf*n 1]));


Valid_id = datasample(1:n,0.2*n,'Replace',false);
Train_id = setdiff(1:n,Valid_id);

C_train = C(Train_id,:);
C_valid = C(Valid_id,:);

% Oringal Setting:
Y_train = Y(Train_id,:);
Y_valid = Y(Valid_id,:);

%%



sTheta = [3 3 3 4]; % size of theta
m = prod(sTheta); % length of theta
stepsize = 5;
winiguess = randn([nc*(nf*sTheta(4)/sTheta(3))+m+1 1]);

Niter = 1000;
Etrain = zeros(Niter,1);
Evalid = zeros(Niter,1);
Thetanorm = zeros(Niter,1);
Acctrain = zeros(Niter,1);
Accvalid = zeros(Niter,1);
Iter = 1;

size_batch = 200;

while Iter <= Niter

    
    minbatch = datasample(1:size(Y_train,1),size_batch,'Replace',false);
    Y_train_batch = Y_train(minbatch,:);
    C_train_batch = C_train(minbatch,:);
    
    [Ecurr,dEcurr] = convCouple2DLayerNNObjFun(winiguess,Y_train_batch,C_train_batch,m,sTheta,nImg);

    LineSearch = 0;
    while convCouple2DLayerNNObjFun(winiguess-stepsize*double(dEcurr),Y_train_batch,C_train_batch,m,sTheta,nImg)>= Ecurr
        LineSearch = 1;
        stepsize = stepsize/2;
        if stepsize <= 1e-6
            fprintf('Step Size is zero!\n');
            break
        end
    end

    wnew = winiguess-stepsize*dEcurr;
    
    Thetanorm(Iter) = norm(stepsize*dEcurr)/norm(winiguess);
    %Etrain(Iter) = convCouple2DLayerNNObjFun(wnew,Y_train,C_train,m,sTheta,nImg);
    Etrain(Iter) = convCouple2DLayerNNObjFun(wnew,Y_train_batch,C_train_batch,m,sTheta,nImg);
    Evalid(Iter) = convCouple2DLayerNNObjFun(wnew,Y_valid,C_valid,m,sTheta,nImg);
    
    %Acctrain(Iter) = accur_singleLayerNN(wnew,Y_train,C_train,m);
    %Accvalid(Iter) = accur_singleLayerNN(wnew,Y_valid,C_valid,m);

    fprintf('Train E: %1.2e;\t Valid E: %1.2e;\t ParameterChange: %1.2e\n',Etrain(Iter),Evalid(Iter),Thetanorm(Iter))
    %fprintf('Train Loss: %1.2e;\t Valid Loss: %1.2e;\t Iter: %1e\n',Acctrain(Iter),Accvalid(Iter),Iter)

    if LineSearch == 0
        stepsize = stepsize*1.5;
    end

    winiguess = wnew;
    Iter = Iter + 1;
end

subplot(1,3,1)
plot(Etrain,'r-')
hold on;
plot(Evalid,'b-')

subplot(1,3,2)
plot(Acctrain,'r-')
hold on;
plot(Accvalid,'b-')

subplot(1,3,3)
plot(Thetanorm)

