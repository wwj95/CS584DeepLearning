% One Layer Network f(YK+b)W 

clear;

% Load data

[Y, C] = setupPeaks();
MaxIter = 100;
n  = size(Y,1);
nc = size(C,2);
Y = [Y ones(n,1)];
nf = size(Y,2);

Valid_id = datasample(1:n,0.2*n,'Replace',false);
Train_id = setdiff(1:n,Valid_id);

C_train = C(Train_id,:);
C_valid = C(Valid_id,:);


% Oringal Setting:
Y_train = Y(Train_id,:);
Y_valid = Y(Valid_id,:);

%%

m = 20;
stepsize = 5;
winiguess = randn([nc*m+m*nf+1 1]);

Niter = 1000;
Etrain = zeros(Niter,1);
Evalid = zeros(Niter,1);
Thetanorm = zeros(Niter,1);
Acctrain = zeros(Niter,1);
Accvalid = zeros(Niter,1);
Iter = 1;

while Iter <= Niter

    [Ecurr,dEcurr,~] = singleLayerNNObjFun(winiguess,Y_train,C_train,m);

    LineSearch = 0;
    while singleLayerNNObjFun(winiguess-stepsize*dEcurr,Y_train,C_train,m)>= Ecurr
        LineSearch = 1;
        stepsize = stepsize/2;
        if stepsize <= 1e-6
            fprintf('Step Size is zero!\n');
            break
        end
    end

    wnew = winiguess-stepsize*dEcurr;
    
    Thetanorm(Iter) = norm(stepsize*dEcurr)/norm(winiguess);
    Etrain(Iter) = singleLayerNNObjFun(wnew,Y_train,C_train,m);
    Evalid(Iter) = singleLayerNNObjFun(wnew,Y_valid,C_valid,m);
    
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


%%%


% 1D Conn for Peaks  f(Y(theta)K+b)W 

clear;

% Load data

[Y, C] = setupPeaks();
MaxIter = 100;
n  = size(Y,1);
nc = size(C,2);
Y = Y * randn([2 20]);
nf = size(Y,2);

Valid_id = datasample(1:n,0.2*n,'Replace',false);
Train_id = setdiff(1:n,Valid_id);

C_train = C(Train_id,:);
C_valid = C(Valid_id,:);


% Oringal Setting:
Y_train = Y(Train_id,:);
Y_valid = Y(Valid_id,:);

%%

m = 5;
stepsize = 5;
winiguess = randn([nc*nf+m+1 1]);

Niter = 1000;
Etrain = zeros(Niter,1);
Evalid = zeros(Niter,1);
Thetanorm = zeros(Niter,1);
Acctrain = zeros(Niter,1);
Accvalid = zeros(Niter,1);
Iter = 1;

while Iter <= Niter

    [Ecurr,dEcurr] = conn1DLayerNNObjFun(winiguess,Y_train,C_train,m);

    LineSearch = 0;
    while conn1DLayerNNObjFun(winiguess-stepsize*dEcurr,Y_train,C_train,m)>= Ecurr
        LineSearch = 1;
        stepsize = stepsize/2;
        if stepsize <= 1e-6
            fprintf('Step Size is zero!\n');
            break
        end
    end

    wnew = winiguess-stepsize*dEcurr;
    
    Thetanorm(Iter) = norm(stepsize*dEcurr)/norm(winiguess);
    Etrain(Iter) = conn1DLayerNNObjFun(wnew,Y_train,C_train,m);
    Evalid(Iter) = conn1DLayerNNObjFun(wnew,Y_valid,C_valid,m);
    
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



%%

% 2D Conn for Peaks  f(Y(theta)K+b)W 

clear;

% Load data

[Y, C] = setupPeaks();
MaxIter = 100;
n  = size(Y,1);
nc = size(C,2);

Y = Y * randn([2 100]);
nImg = [10 10]; % size of figure

nf = size(Y,2);

Valid_id = datasample(1:n,0.2*n,'Replace',false);
Train_id = setdiff(1:n,Valid_id);

C_train = C(Train_id,:);
C_valid = C(Valid_id,:);


% Oringal Setting:
Y_train = Y(Train_id,:);
Y_valid = Y(Valid_id,:);

%%

m = 9; % length of theta
sTheta = [3 3]; % size of theta
stepsize = 5;
winiguess = randn([nc*nf+m+1 1]);

Niter = 1000;
Etrain = zeros(Niter,1);
Evalid = zeros(Niter,1);
Thetanorm = zeros(Niter,1);
Acctrain = zeros(Niter,1);
Accvalid = zeros(Niter,1);
Iter = 1;

while Iter <= Niter

    [Ecurr,dEcurr] = conn2DLayerNNObjFun(winiguess,Y_train,C_train,m,sTheta,nImg);

    LineSearch = 0;
    while conn2DLayerNNObjFun(winiguess-stepsize*dEcurr,Y_train,C_train,m,sTheta,nImg)>= Ecurr
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





