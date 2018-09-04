function w = cgls(Y,c,k)
% Conjugate Gradient for Least Squares
%
% solves min_w |Y*w-c|^2
%

n = size(Y,2);
w = zeros(n,1);
d = Y'*c;   r = c;
normr2 = d'*d;
for j=1:k
    Ad = Y*d; alpha = normr2/(Ad'*Ad); w =w+alpha*d;
    r = r - alpha*Ad;
    s = Y'*r;
    normr2New = s'*s;
    beta = normr2New/normr2;
    normr2 = normr2New;
    d = s + beta*d;
end