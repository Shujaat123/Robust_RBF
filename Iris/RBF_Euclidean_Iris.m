%% (Robust-RBF): Multi-Kernel Fusion for RBF NN
% Author: SHUJAAT KHAN, shujaat123@gmail.com
clc
clear all
close all
load Data\Iris_Processed.csv  % Gene Expression Micro Array Data for Leukemia cancer by Golub 

neg_ind=find(Iris_Processed(:,end)==1);
neu_ind=find(Iris_Processed(:,end)==0);
pos_ind=find(Iris_Processed(:,end)==-1);

train_ind=[neg_ind(1:40);neu_ind(1:40);pos_ind(1:40)];
test_ind=[neg_ind(41:end);neu_ind(41:end);pos_ind(41:end)];

train_data=Iris_Processed(train_ind,1:end-1);
test_data=Iris_Processed(test_ind,1:end-1);
class_train=Iris_Processed(train_ind,end)';
class_test=Iris_Processed(test_ind,end)';

feature_length=4;  % Number of features used to design the classifier

train_data;
class_train;

[m1 ~] = size(train_data);

epoch=2000;
eta=5e-3;
interval=0.5;
beeta=2*interval;
runs=100;
[c] = subclust(train_data,0.2); 
n1 = size(c,1);

I=zeros(1,epoch);
alpha_final=0;
y_test_final=0;
for run=1:runs

w=randn(1,n1);
b=randn();
% I=0;

for k=1:epoch
%     I(k)=0;
    ind=1:m1;
    
    for i1=1:m1
        for i2=1:n1
            phi(1,i2)=exp((-(norm(train_data(ind(i1),:)-c(i2,:))^2))/beeta^2);
        end
        y_train(i1)=w*phi(1,:)'+b;
        d(i1)=class_train(ind(i1));
        e=d(i1)-y_train(i1);
        I(k)=I(k)+e*e';   %%% Objective Function
          
        w=w+eta*e*phi(1,:);
        b=b+eta*e;
    end

end
Training_Accuracy(run)=100*(1-(length(find((round(y_train)-class_train(ind))~=0))/length(y_train)));

%% Testing
P=test_data;
f=class_test;

[m2 ~] = size(P);
for i1=1:m2
    for i2=1:n1
        phi(1,i2)=exp((-(norm(test_data(ind(i1),:)-c(i2,:))^2))/beeta^2);
    end
    y_test(i1)=w*phi(1,:)' + b;
end

y_test_final=y_test_final+y_test;
Testing_Accuracy(run)=100*(1-(length(find((round(y_test)-class_test)~=0))/length(y_test)));
end
I=I./(runs*m1);
alpha_final=[0 1];
y_test_final=y_test_final./runs;

save('Results\I2.mat','I','eta','f','y_test_final','Training_Accuracy','Testing_Accuracy','alpha_final');

Graphs

