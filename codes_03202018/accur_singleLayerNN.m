function[acc] =  accur_singleLayerNN(wnew,Y,C,m)

    [n,nf] = size(Y);
    nc     = size(C,2);

    x = wnew(:);
    K = reshape(x(1:nf*m),nf,m);
    b = x(nf*m+1);
    W = reshape(x(nf*m+2:end),[],nc);
    S_est = tanh(Y*K+b)*W;
    S_est = S_est - (max(S_est,[],2))* ones(1,nc);
    Cest = diag(1./( exp(S_est)*ones(nc,1) ))*exp(S_est);
    
    acc = sum(abs(find(C' == max(C')) - find(Cest' == max(Cest'))))/n;
end