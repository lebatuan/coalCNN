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
   N=10; 
   AF='sig'; 
   yincengnum=4;  
    t0=cputime;

      IW=rand(N,Pm)*2 - 1;
  % [Q, IW] = qr(IW); 
  % IW=-sqrt(6./(Pn+Pn))+2*sqrt(6./(Pn+Pn))*rand(N,Pm);  
      B=rand(N,1);
   
      
     
[IW,B,T_IW,T_B,TLW,J,IN,TF,TYPE] = elm1(P_train,T_train,yincengnum,N,AF,1,IW,B);


T_sim_2 = elm2(P_test,IW,B,T_IW,T_B,TLW,J,IN,TF,TYPE);   

 Tn  = T_test;
k1 = length(find(T_sim_2 == Tn));
n1 = length(Tn);
Accuracy_1 = k1 / n1;

t1=(cputime-t0);
disp(['Accuracy = ' num2str(Accuracy_1*100) '%' ])


 
