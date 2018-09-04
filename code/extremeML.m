function [Y_new] = extreamML(Y,p,num, func)
    rng(num)
    k=randn(size(Y,2),p);
    b=randn(size(Y,1),p);
    Y_new=func(Y*k+b);
    
end
