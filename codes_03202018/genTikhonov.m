function[R,dR,d2R] = genTikhonov(W,param)
%[R,dR,d2R] = genTikhonov(W,param)
%
%  R(W) = (h/2)* | L*W|_F^2
% 
% where the scalar h adapts for mesh size and L is a matrix (e.g.,
% differential operator)
% 
% Inputs:
%   W      - weights, either as nf x nc matrix or vector (preferred in
%            optimization)
%   param -  struct whose fields control the type of regularizer. 
%            Fields include:
%               nc - number of classes
%               h  - mesh-size, e.g., pixel size in 2D
%               L  - matrix

if nargin==0
    testGenTikhonov
    return;
end

% Your code here!