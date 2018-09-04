function [E,J,dE,d2Emv]=softMax(W,Y,C,alpha)

%%%%put 1/n for entropy
% Your code from before
% alpha regularization parameters
sizetemp=size(Y);
n=sizetemp(1);
sizetemp2=size(C);
nc=sizetemp2(2);
W = reshape(W, [], nc);
if size(W,1)~=size(Y,2)
       Y = [Y ones(size(Y,1), 1)];
 end

YW=Y*W;
enc=ones(nc,1);
en=ones(n,1);
Scale=max(YW,[],2)*enc'; %scale to avoid overflow
S=YW-Scale;
E=((-trace(C'*S)+en'*log(exp(S)*enc))).*1/n;
if nargout >1
J=((-trace(C'*S)+en'*log(exp(S)*enc))).*1/n+alpha/2*trace(W'*W);
if nargout >2
%Your code for gradient here
dE=(Y'*(-C+exp(S).*((1./(exp(S)*enc))*enc'))).*1/n+alpha*W;
dE = dE(:);
end
if nargout>3
d2Emv=@(V) reshape(Y'*((exp(S)./((exp(S)*enc)*enc')).*(Y*reshape(V,[],nc)))-Y'*((exp(S)./((exp(S)*enc).^2*enc')).*((exp(S).*(Y*reshape(V,[],nc)))*(enc)*enc')),[],1).*(1/n)+alpha*reshape(V,[],1);
end
end
