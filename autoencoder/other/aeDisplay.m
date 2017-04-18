function aeDisplay(ae, mid, iter)

subplot(2,2,1);
display_network(ae.w1');
subplot(2,2,2);
display_network(ae.w2);

subplot(2,2,3);
plot([1:i],mloss(1:i), 'b');
axis([0 numEpochs 0 ceil(max(mloss))]);
subplot(2,2,4);
plot([1:i],mloss(1:i), 'b');

end