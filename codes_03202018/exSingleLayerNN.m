close all; clear all; clc;

rng(20)
n  = 50; nf = 50; nc = 3; m  = 40;
Wtrue = randn(m,nc);
Ktrue = randn(nf,m);
btrue = .1;

Y     = randn(n,nf);
Cobs  = exp(singleLayer(Ktrue,btrue,Y)*Wtrue);
Cobs  = Cobs./sum(Cobs,2);

%%
dW = randn(m,nc);
dK = randn(nf,m);

[tW,tK] = ndgrid(linspace(-1,1,41));
E = 0*tW;
for i=1:size(tW,1)
    for j=1:size(tW,2)
        Zt = singleLayer(Ktrue+tK(i,j)*dK,btrue,Y);
        E(i,j)=softMax(Wtrue+tW(i,j)*dW,Zt,Cobs);
        
    end
end

%%
figure(1); clf;
contour(tW,tK,E,'lineWidth',2)
xlabel('W + tW*dW')
ylabel('K + tK*dK')
set(gca,'FontSize',20)

%%
figure(2); clf;
surf(tW,tK,E,'lineWidth',2)
xlabel('W + tW*dW')
ylabel('K + tK*dK')
set(gca,'FontSize',20)