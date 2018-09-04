a = 3;
b = 2;

Y = randn(500,2);
C = a*Y(:,1) + b*Y(:,2) +2;
C(C>0) = 1; C(C<0) = 0;
C  = [C, 1-C];

W  = [eye(2); ones(1,2)];
W  = W(:);

%% check calls of softmax
[F1,dF1] = softMax(W,C,C);
if not(numel(F1)==1)
    error('first output argument of softMax must be a scalar')
end
if any(size(dF1)~=size(W))
    error('size of gradient and W must match')
end

[F2,dF2] = softMax(1e4*W,C,C);
if isinf(F2) || isnan(F2)
    error('Likely an overflow in softMax ')
end
if abs(F2)>1e-9
    error('loss should be around zero')
end

[F3,dF3] = softMax(W,[1e4*C; 1 0],[C; 1 0]);
if isinf(F3) || isnan(F3)
    error('Likely an underflow in softMax ')
end

%% check derivatives and Hessian
W0 = randn(size(W));
dW = randn(size(W));

[F,dF,d2F] = softMax(W0,Y,C);
dFdW = dF'*dW;
d2FdW = dW'*d2F(dW);
% dF = dF + 1e-2*randn(size(dF));
err    = zeros(30,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = softMax(W0+h*dW,Y,C);
    
    err(k,:) = [h, norm(F-Ft), norm(F+h*dFdW-Ft), norm(F+h*dFdW+h^2/2*d2FdW-Ft)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
hold on;
loglog(err(:,1), err(:,4),'-k','linewidth',3);
legend('E0','E1','E2');

%% check derivatives and Hessian
Y0 = randn(size(Y));
dY = randn(size(Y));

[F,dF,d2F,dFY,d2FY] = softMax(W,Y0,C);
dFdY = dFY'*dY(:);
d2FdY = dY(:)'*d2FY(dY);
% dF = dF + 1e-2*randn(size(dF));
err    = zeros(30,4);
for k=1:size(err,1)
    h = 2^(-k);
    Ft = softMax(W,Y0+h*dY,C);
    
    err(k,:) = [h, norm(F-Ft), norm(F+h*dFdY-Ft), norm(F+h*dFdY+h^2/2*d2FdY-Ft)];
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\tE1=%1.2e\n',err(k,:))
end

figure; clf;
loglog(err(:,1),err(:,2),'-b','linewidth',3);
hold on;
loglog(err(:,1), err(:,3),'-r','linewidth',3);
hold on;
loglog(err(:,1), err(:,4),'-k','linewidth',3);
legend('E0','E1','E2');