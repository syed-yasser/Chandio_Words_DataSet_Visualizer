clear all
close all
clc


% load('Yasser_SyntheticDataSet_4.212K_GoogleNet_ResNet50_FeatureTargetsSaver.mat')
% above .mat file is made by code-file "Features_ResNet50_Googlenet_Extractor_For_Synthetic_4212_K_im.m" 
%load('Yasser_Urdu_OutDoor_DataSet_GoogleNet_ResNet50_FeatureTargetsSaver_v2.mat');
load('Yasser_Urdu_Chandio2020_DataSet_GoogleNet_ResNet50_FeatureTargetsSaver_v1.mat','XTrainGoogleNetTargets','XTestGoogleNetTargets');

NewTargets=[XTrainGoogleNetTargets XTestGoogleNetTargets];
documents = tokenizedDocument(NewTargets);
figure
wordcloud(documents);
title ('Simple Words Cloud ...');

bag = bagOfWords(documents);
figure,
wordcloud(bag); title ('bag of Words Cloud ...');

% bag = bagOfNgrams(documents)
% figure 
% wordcloud(bag);

bag = bagOfNgrams(documents,'NgramLengths',[2 3 4]);
tbl2=topkngrams(bag,10,'NGramLengths',2);
tbl3=topkngrams(bag,10,'NGramLengths',3);
tbl4=topkngrams(bag,10,'NGramLengths',4);


bag = bagOfNgrams(documents,'NgramLengths',1);
figure 
wordcloud(bag); title ('Top-1 Ngrams Words Cloud ...');

bag = bagOfNgrams(documents,'NgramLengths',2);
figure 
wordcloud(bag); title ('Top-2 Ngrams Words Cloud ...');
bag = bagOfNgrams(documents,'NgramLengths',3);
figure 
wordcloud(bag); title ('Top-3 Ngrams Words Cloud ...');

bag = bagOfNgrams(documents,'NgramLengths',4);
figure 
wordcloud(bag);title ('Top-4 Ngrams Words Cloud ...');

%%
T = wordCloudCounts(NewTargets);
head(T)




%%
mdl = fitlda(bag,20,'Verbose',0);
idx_y=1
figure
for topicIdx = 1:4
    subplot(2,2,topicIdx)
    wordcloud(mdl,topicIdx+idx_y);
    title("Topic: " + num2str(topicIdx+idx_y))
end

NoOfWordsForFrequency=20;
x=categorical(T.Word(1:NoOfWordsForFrequency));
y=(T.Count(1:NoOfWordsForFrequency))
bar(x,y,'stacked');

figure
b = bar(x,y,'FaceColor','flat');
for k = 1:size(y,1)
    b(k).CData = k;
end