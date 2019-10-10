function point_plot(point,data)
% load('J3.mat')
L1MAG=data.L1MAG;
L2MAG=data.L2MAG;
L3MAG=data.L3MAG;
C1MAG=data.C1MAG;
C2MAG=data.C2MAG;
C3MAG=data.C3MAG;
PA=data.PA;
PB=data.PB;
PC=data.PC;
QA=data.QA;
QB=data.QB;
QC=data.QC;
start=point*20-120;
endd=point*20+120;


subplot(221);
plot(C1MAG(start : endd));
hold on
plot(C2MAG(start : endd)); 
hold on
plot(C3MAG(start : endd)); title('I')

subplot(222); 
plot(L1MAG(start : endd));
hold on
plot(L2MAG(start : endd)); 
hold on
plot(L3MAG(start : endd)); title('V')
   
subplot(223); plot(PA(start : endd)); 
plot(PA(start : endd));
hold on
plot(PB(start : endd)); 
hold on
plot(PC(start : endd));title('P')
  
subplot(224); 
plot(QA(start : endd));
hold on
plot(QB(start : endd)); 
hold on
plot(QC(start : endd));
title('Q')
end