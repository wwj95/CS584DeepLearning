% 2D Conn for MNIST  f(Y(theta)K+b)W 

clear;
addpath('/Users/johnz/Desktop/Coursework/MATH Numerical Methods for DL/codes_03202018/')
addpath('/Users/johnz/')
addpath('/Users/johnz/Documents/MATLAB/')



% Load data


Y = loadMNISTImages('/Users/johnz/Desktop/Coursework/MATH Numerical Methods for DL/Project/hw1/train-images-idx3-ubyte')';
c = loadMNISTLabels('/Users/johnz/Desktop/Coursework/MATH Numerical Methods for DL/Project/hw1/train-labels-idx1-ubyte');
nex = length(c);
C = full(sparse(1:nex,c+1,ones(nex,1),nex,10));

% 2D Conn for Peaks  f(Y(theta)K+b)W 

n  = size(Y,1);
nc = size(C,2);
nImg = [28 28]; % size of figure

nf = size(Y,2);

Valid_id = datasample(1:n,0.2*n,'Replace',false);
Train_id = setdiff(1:n,Valid_id);

C_train = C(Train_id,:);
C_valid = C(Valid_id,:);


% Oringal Setting:
Y_train = Y(Train_id,:);
Y_valid = Y(Valid_id,:);

%%

%m = 25; % length of theta
sTheta = [5 5]; % size of theta
m = prod(sTheta);
stepsize = 5;
winiguess = randn([nc*nf+m+1 1]);

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
    
    [Ecurr,dEcurr] = conn2DLayerNNObjFun(winiguess,Y_train_batch,C_train_batch,m,sTheta,nImg);

    LineSearch = 0;
    while conn2DLayerNNObjFun(winiguess-stepsize*dEcurr,Y_train_batch,C_train_batch,m,sTheta,nImg)>= Ecurr
        LineSearch = 1;
        stepsize = stepsize/2;
        if stepsize <= 1e-6
            fprintf('Step Size is zero!\n');
            break
        end
    end

    wnew = winiguess-stepsize*dEcurr;
    
    Thetanorm(Iter) = norm(stepsize*dEcurr)/norm(winiguess);
    Etrain(Iter) = conn2DLayerNNObjFun(wnew,Y_train,C_train,m,sTheta,nImg);
    Evalid(Iter) = conn2DLayerNNObjFun(wnew,Y_valid,C_valid,m,sTheta,nImg);
    
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


Wbest = reshape(wnew(m+2:end),nf,nc);

for i = 1:10
    subplot(2,5,i)
    imagesc(reshape(Wbest(:,i),[28 28]))
end




