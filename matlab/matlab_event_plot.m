clc
clear all

%load the events for July third
events=csvread('anoms_July_03.csv');

%load the main data for July third which contains
%L1MAG,L2MAG,L3MAG,C1MAG,C2MAG,C3MAG,PA,PB,PC,QA,QB,QC
data=load('J3.mat');

%define the point and plot the main data from the point
%here point is the nth detected event
point=500
figure
%we should pass the point and the main data to the function
point_plot(events(point),data)
