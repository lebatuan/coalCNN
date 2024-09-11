function [IW,B,T_IW,T_B,TLW,J,N,TF,TYPE] = elm1(P,T,J,N,TF,TYPE,IW10,B10)
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)  
% T   - Output Matrix of Training Set (S*Q)  
% N   - Number of Hidden Neurons (default = Q)  
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMPREDICT
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 3
    N = size(P,2);
end
if nargin < 4
    TF = 'sig';
end
if nargin < 5
    TYPE = 0;
end
if size(P,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end

[R,Q] = size(P);
if TYPE  == 1
    T  = ind2vec(T); 
end
[S,Q] = size(T);

% Randomly Generate the Input Weight Matrix
%IW = rand(N,R) * 2 - 1;  % N*R
IW=IW10;
% Randomly Generate the Bias Matrix
%B = rand(N,1);
B=B10;
%%%%%%%  BiasMatrix = repmat(B,1,Q);  % N*Q      
ind = ones(1,Q);
BiasMatrix = B(:,ind);                       
% Calculate the Layer Output Matrix H
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));  % N*Q    pinv(H')为N*Q; 
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

T_B = []; T_IW = [];
for i = 1:1:J-1
     LW = pinv(H')*T';
  % if(N<Q)
    %  LW = (pinv(rand*eye(Q,Q)+H'*H)*H')'* T';  % N*S   
 %  elseif(N>Q)
  %    LW = (H'*pinv(rand*eye(N,N)+H*H'))'* T';  % N*S
  % end
    H_iE = (T'*pinv(LW))';  % N*Q
    M = [ind;H];  
  
    switch TF
        case 'sig'
            WH = (log(H_iE)-log(ones(size(H_iE))-H_iE))*pinv(M);
    end
  
    B1=WH(:,1);  % N*Q  
    IW1=WH(:,2:size(H,1)+1);  % N*N
    TtempH = IW1*H+B1(:,ind);
  
    switch TF
        case 'sig'
              H = 1./(1+exp(-TtempH));
        case 'sin'
              H = sin(TtempH);
        case 'hardlim'
              H = hardlim(TtempH);
    end
   
    
    T_B = [T_B B1];
    T_IW = [T_IW IW1];
end

  TLW = pinv(H') * T';
%if(N<Q)
 %  TLW = (pinv(rand*eye(Q,Q)+H'*H)*H')'* T';  % N*S   
%  elseif(N>Q)
%    TLW = (H'*pinv(rand*eye(N,N)+H*H'))'* T';  % N*S
%end


