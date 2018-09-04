function [w,history] = sdls(Y,c,k)

if nargin==0
    a = 1e-1;
   Y  = [1 1+1*a; 1 1+2*a; 1 1+3*a];
   wt =[1; 1.2];
   c  = Y*wt + randn(3,1)/10;
   
   [w,h] = sdls(Y,c,20);
   figure;
   semilogy(h);
   return;
end

n = size(Y,2);
w = zeros(n,1);
history = zeros(k,1);
for j=1:k
    r  = Y*w-c;
    gc = Y'*r;
    history(j) = norm(gc);
    alpha = r'*(Y*gc) / norm(Y*gc)^2;
    w = w- alpha*gc;  
end