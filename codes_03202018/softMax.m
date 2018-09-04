function[E,dEW,d2EW,dEY,d2EY] = softMax(W,Y,C)
%[E] = softMax(W,Y,C)
% 
% Evaluates cross-entropy function for multinomial classification
%

if nargin == 0
   runMinExample;
   return
end
W = reshape(W,[],size(C,2)); addBias = false;
if size(W,1)==size(Y,2)+1
    addBias = true;
    Y = [Y, ones(size(Y,1),1)];
end
n = size(Y,1);
nc = size(C,2);

% the linear model
S = Y*W;

% accuracy:
 

% make sure that the largest number in every row is 0
s = max(S,[],2);
S = S-s;

    

% The cross entropy
expS = exp(S);
sS   = sum(expS,2);

E = -C(:)'*S(:) +  sum(log(sS)); 
E = E/n;

if nargout > 1
    dES  = -C + expS .* 1./sS;
    dEW  = (Y'*dES)/n;
    dEW = dEW(:);
end

if nargout>2
    matW = @(v) reshape(v,[],nc); % reshape vector into same size of W
    vec = @(V) V(:);
    
    d2E1 = @(v) (1/n) * (Y'*( (expS./sS) .* (Y*matW(v))));
    d2E2 = @(v) (-1/n) *(Y'*  (expS.* ((1./sS.^2) .* sum(expS.*(Y*matW(v)),2))));
    d2EW  = @(v) vec(d2E1(v) + d2E2(v)) + 1e-5*vec(v);
    dEW = dEW(:);
end

if addBias
    W = W(1:end-1,:);
end
if nargout > 3
    dEY  = dES*(W'/n);
    dEY  = dEY(:);
end

if nargout>4
    matY = @(v) reshape(v,n,[]);
    
    d2EY1 = @(v) (1/n) * (( (expS./sS) .* (matY(v)*W))*W');
    d2EY2 = @(v) (-1/n) *( (expS.* ((1./sS.^2) .* sum(expS.*(matY(v)*W),2)))*W');
    d2EY  = @(v) vec(d2EY1(v) + d2EY2(v)) + 1e-5*vec(v);
end

end

function runMinExample

vec = @(x) x(:);
nex = 100;
Y = hilb(500)*255;
Y = Y(1:nex,:);
C = ones(nex,3);
C = C./sum(C,2);
b = 0;
W = hilb(501);
W = W(:,1:3);
E = softMax(W,Y,C);
[E,dE,d2E] = softMax(W,Y,C);

h = 1;
rho = zeros(20,3);
dW = randn(size(W));
for i=1:20
    E1 = softMax(W+h*dW,Y,C);
    t  = abs(E1-E);
    t1 = abs(E1-E-h*dE(:)'*dW(:));
    t2 = abs(E1-E-h*dE(:)'*dW(:) - h^2/2 * dW(:)'*vec(d2E(dW)));
    
    fprintf('%3.2e   %3.2e   %3.2e\n',t,t1,t2)
    
    rho(i,1) = abs(E1-E);
    rho(i,2) = abs(E1-E-h*dE(:)'*dW(:));
    rho(i,3) = t2;
    h = h/2;
end

rho(2:end,:)./rho(1:end-1,:);

end