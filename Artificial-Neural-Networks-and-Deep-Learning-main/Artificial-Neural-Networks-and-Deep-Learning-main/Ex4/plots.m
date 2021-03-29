%% Artificial Neural Networks : Exercise 4

batch= 1:200:20000;
figure;
subplot(1,2,1);
plot(batch,a(:,1),'-*');
hold on; 
plot(batch,a(:,3),'-o');
title('Discriminator and Generator Loss Curves')
xlabel('Batch')
ylabel('Value')
legend('D Loss','G Loss')
hold off;

subplot(1,2,2)
plot(batch,a(:,2),'-*');
hold on; 
plot(batch,a(:,4),'-o');
title('Discriminator and Generator Accuracy Curves')
xlabel('Batch')
ylabel('Value')
legend('D Acc','G Acc')