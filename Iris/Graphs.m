%% (Robust-RBF): Multi-Kernel Fusion for RBF NN
% Author: SHUJAAT KHAN, shujaat123@gmail.com
close all
figure
fsize=13;
fmt='eps';
load Results\I1
plot(10*log10(I),'-.','linewidth',2)
hold on
load Results\I2
plot(10*log10(I),'--r','linewidth',2)
load Results\I3
plot(10*log10(I),'--k','linewidth',2)
load Results\I4
plot(10*log10(I),'g','linewidth',2)
load Results\I5
plot(10*log10(I),'m','linewidth',2)
% xlabel('Number of epochs');
% ylabel('Cost Function');
% grid on
% title(sprintf('Cost function vs epochs at learning rate(eta) = %f',eta));
h=legend('Cosine kernel based RBF','Euclidean kernel based RBF','Manual fusion of the two kernels','Adaptive fusion of the two kernels','Robust fusion of the two kernels','Location','Best');
grid minor
xlabel('Epochs','FontSize',fsize);
ylabel('MSE (dB)','FontSize',fsize);
set(h,'FontSize',12)
saveas(gcf,strcat('Figures\ClassificationMSE.',fmt),'psc2')
saveas(gcf,strcat('Figures\ClassificationMSE.png'),'png')
set(gca,'FontSize',13)

% figure
% load Results\I4
% alpha=[0.5 0.5;alpha(1:end-1,:)];
% plot(alpha(:,1),'--r','Linewidth',3);
% hold on
% plot(alpha(:,2),'--b','Linewidth',3);
% load Results\I5
% alpha=[0.5 0.5;alpha(1:end-1,:)];
% plot(alpha(:,1),'r','Linewidth',3);
% hold on
% plot(alpha(:,2),'b','Linewidth',3);
% 
% hold off
% h=legend('$\alpha1$ : Cosine kernel','$\alpha2$ : Euclidean kernel','$\frac{||W_1||}{||W_1||+||W_2||}$ : Cosine weight ratio','$\frac{||W_2||}{||W_1||+||W_2||}$ : Euclidean weight ratio','Location','Best');
% set(h,'Interpreter','latex','FontSize',12)
% grid minor
% xlabel('Epochs','FontSize',fsize);
% ylabel('Mixing parameter value','FontSize',fsize);
% 
% 
% saveas(gcf,strcat('ClassificationALPHA.',fmt),'psc2')
% saveas(gcf,strcat('ClassificationALPHA.png'),'png')
% set(gca,'FontSize',13)
% figure
% load Results\I1
% plot(f,'-+');
% hold on;
% plot((((y_test_final))),'*-b');
% load Results\I2
% plot((((y_test_final))),'o-r');
% load Results\I3
% plot((((y_test_final))),'*-k');
% load Results\I4
% plot((((y_test_final))),'.-g');
% load Results\I5
% plot((((y_test_final))),'.-m');
% h=legend('Cosine kernel based RBF','Euclidean kernel based RBF','Manual fusion of the two kernels','Adaptive fusion of the two kernels','Dynamic fusion of the two kernels','Location','Best');
% grid minor
% xlabel('Test Samples','FontSize',fsize);
% ylabel('Output score','FontSize',fsize);
% set(h,'FontSize',12)
% saveas(gcf,strcat('ClassificationOutput.',fmt),'psc2')
% saveas(gcf,strcat('ClassificationOutput.png'),'png')
% set(gca,'FontSize',13)

load Results\I1
% Testing_Accuracy=100*(1-(length(find((round(abs(y))-f)~=0))/length(y)));
% [val1 pos1]=min(I);
ResultsCK=[mean(Training_Accuracy) mean(Testing_Accuracy) I(end) sum((f-y_test_final).^2) alpha_final]
load Results\I2
% [val2 pos2]=min(I);
% Testing_Accuracy=100*(1-(length(find((round(y)-f)~=0))/length(y)));
ResultsEK=[mean(Training_Accuracy) mean(Testing_Accuracy) I(end) sum((f-y_test_final).^2) alpha_final]
load Results\I3
% [val3 pos3]=min(I);
% Testing_Accuracy=100*(1-(length(find((round(y)-f)~=0))/length(y)));
ResultsNK=[mean(Training_Accuracy) mean(Testing_Accuracy) I(end) sum((f-y_test_final).^2) alpha_final]
load Results\I4
% [val4 pos4]=min(I);
% Testing_Accuracy=100*(1-(length(find((round(y)-f)~=0))/length(y)));
ResultsAK=[mean(Training_Accuracy) mean(Testing_Accuracy) I(end) sum((f-y_test_final).^2) alpha_final]
load Results\I5
% [val5 pos5]=min(I);
% Testing_Accuracy=100*(1-(length(find((round(y)-f)~=0))/length(y)));
ResultsAFK=[mean(Training_Accuracy) mean(Testing_Accuracy) I(end) sum((f-y_test_final).^2) alpha_final]
Results=[ResultsCK;ResultsEK;ResultsNK;ResultsAK;ResultsAFK]

