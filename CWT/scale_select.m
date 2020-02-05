%% clear all/ close figs
close all
clear
clc

%% default paramters
Participant = 32;
Video = 40;
Channel = 32;
Fs = 128;
Time = 63;
addpath('C:\Users\75196\Documents\GitHub\SchoolCourse\PatternRec\EmotionRec\deap\data_preprocessed_matlab')

%% set parameters
frameNum = 10;
totalScale = 64;
wname = 'db4';

%%

for participant = 1:Participant
    fprintf('\nworking on file number %d:\n', participant);
    if(participant<10)
        myfilename = sprintf('s0%d.mat', participant);
    else
        myfilename = sprintf('s%d.mat', participant);
    end
    load(myfilename);
    
    for video=1:Video
        
        %fprintf(filename);
        sumEER = zeros(totalScale,32);
        datastart=128*3;
        datalength=8064-128*3;
        
        
        for channel = 1:32
            data1=zeros(1,8064-datastart);
            for ii =1:datalength
                data1(1,ii)=data(video,channel,ii+datastart);
            end
            
            % decompose into wavelets
            % set scales
            f = 1:totalScale;
            f = Fs/totalScale * f;
            wcf = centfrq(wname);
            scal =  Fs * wcf./ f;
            
            coefs = cwt(data1, 1:64, wname);
            
            % Figures
            % 3d Figure
            %surf(coefs);shading interp;
            energy = abs(coefs.*coefs);
            scaleEnergy = sum(energy,2);
            s = repmat(scaleEnergy,1,datalength);
            p = energy./s;
            entropy = -p.*log(p);
            scaleEntropy = sum(entropy,2);
            EER = scaleEnergy./scaleEntropy;
            sumEER(:,channel) = EER;
            
            
           
            
        end
        
        fprintf('.');    
        

    end %the testcase loop
end %the file loop
%%
%pcolor(sumEER)
%plot(EER);
plot(mean(sumEER,2),'*');
xlabel('Energy/Entropy');
ylabel('Scales')
title('Average EER over All EEG Channels');
[sorted, index] = sort(mean(sumEER,2),'descend');
index(1:32)