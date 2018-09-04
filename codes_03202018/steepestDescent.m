function W = steepestDescent(E,W,param)
% Inputs:
%     E  - function that provides value and gradient
%     W  - starting guess
%  param - struct with parameters

if nargin==0
   E = @Rosenbrock;
   W = [4;2];
   param = struct('maxIter',10000,'maxStep',1);
   W = feval(mfilename,E,W,param);
   
   fprintf('numerical solution: W = [%1.4f, %1.4f]\n',W);
   
   [X,Y] = ndgrid(linspace(-1,2,101));
   Eval = reshape(E([X(:),Y(:)]'),size(X));
   figure; 
   contour(X,Y,Eval,200);
   return
end

mu        = param.maxStep; % max step size
maxIter   = param.maxIter; % max number of iterations

for i=1:maxIter
    % your code here
end

   

function [f,df,d2f] = Rosenbrock(x)
x = reshape(x,2,[]);
f = (1-x(1,:)).^2 + 100*(x(2,:) - (x(1,:)).^2).^2;

if nargout>1 && size(x,2)==1
    df = [2*(x(1)-1) - 400*x(1)*(x(2)-(x(1))^2); ...
        200*(x(2) - (x(1))^2)];
end

if nargout>2 && size(x,2)==1
    n= 2;
    d2f=zeros(n);
    d2f(1,1)=400*(3*x(1)^2-x(2))+2; d2f(1,2)=-400*x(1);
    for j=2:n-1
        d2f(j,j-1)=-400*x(j-1);
        d2f(j,j)=200+400*(3*x(j)^2-x(j+1))+2;
        d2f(j,j+1)=-400*x(j);
    end
    d2f(n,n-1)=-400*x(n-1); d2f(n,n)=200;
end