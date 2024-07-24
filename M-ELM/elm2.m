function Y = elm2(P,IW,B,T_IW,T_B,TLW,J,N,TF,TYPE)
% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMTRAIN
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $
if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end

% Calculate the Layer Output Matrix H
Q = size(P,2);
BiasMatrix = repmat(B,1,Q);  % N*Q
tempH = IW * P + BiasMatrix;
switch TF
       case 'sig'
              H1 = 1 ./ (1 + exp(-tempH));  % N*Q  第一个隐含层输出
       case 'sin'
              H1 = sin(tempH);
       case 'hardlim'
              H1 = hardlim(tempH);
end

i=1;
while i<=J-1
    BiasMatrix1 = repmat(T_B(:,i),1,Q);
    IW1 = T_IW(:,(i-1)*N+1:i*N);
    tempH1 = IW1*H1+BiasMatrix1;
    switch TF
        case 'sig'
            H1 = 1./(1+exp(-tempH1));
        case 'sin'
              H1 = sin(tempH1);
        case 'hardlim'
              H1 = hardlim(tempH1);
    end
    i=i+1;
end

% Calculate the Simulate Output
Y = (H1' * TLW)';

if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y);
end


       
    
