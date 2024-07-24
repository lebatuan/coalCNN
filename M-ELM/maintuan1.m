clear
clc
close all

[train]=xlsread('train_fea');
[test]=xlsread('test_fea');
   P_train =train(:,1:end-1)'/100;
   T_train = train(:,end)';  
   P_test = test(:,1:end-1)'/100; 
   T_test = test(:,end)'; 

   [m,n]=size(test);
   a = randperm(m);  
   b=1.6;
  k1=fix(m/b); 
     P_test1 = test(a(1:k1),1:(end-1))'; 
     T_test1 = test(a(1:k1),end)';    
     P_test2 = test(a(k1+1:m),1:(end-1))'; 
     T_test2 = test(a(k1+1:m),end)';
    
    [Pm,Pn]=size(P_train); 
   N=10; %ELM隐含层节点数
   AF='sig'; %激活函数
   yincengnum=4;  %% 隐含层的数目
    t0=cputime;

      IW=rand(N,Pm)*2 - 1;
  % [Q, IW] = qr(IW); %正交化
  % IW=-sqrt(6./(Pn+Pn))+2*sqrt(6./(Pn+Pn))*rand(N,Pm);  %Xavier权重初始化. xavier权重初始化的作用，使得信号在经过多层神经元后保持在合理的范围（不至于太小或太大）。
      B=rand(N,1);
   
       %% ELM创建/训练
     
[IW,B,T_IW,T_B,TLW,J,IN,TF,TYPE] = elm1(P_train,T_train,yincengnum,N,AF,1,IW,B);

%% ELM仿真测试
T_sim_2 = elm2(P_test,IW,B,T_IW,T_B,TLW,J,IN,TF,TYPE);   % 测试集预测的结果

 Tn  = T_test;
k1 = length(find(T_sim_2 == Tn));
n1 = length(Tn);
Accuracy_1 = k1 / n1;

t1=(cputime-t0);
disp(['测试集正确率Accuracy = ' num2str(Accuracy_1*100) '%' ])


 