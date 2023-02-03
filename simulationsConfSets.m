% This script:
%
% a) randomly generates an nxp design matrix X with independent and identically
% distributed rows, generated from a N(0, Sigma) distribution.
%
% b) randomly generates a linear response Y=a\mathbb{1}+Xb+e where a is an
% intercept term, b is a sparse vector with s non-zero entries and e is
% centred, gaussian noise.
%
% c) constructs various comprehensive models using Cox Reduction (original,
% sample splitting and randomised versions), marginal screening and an
% underturned lasso.
%
% d) constructs a confidence set of models based on each comprehensive
% model.
%
% The quality of each comprehensive model and the corresponding confidence
% set of models is recorded by averaging various factors over Monte Carlo
% repetitions.


% For further details please see: Section S2.1 of Lewis, R. M. and 
% Battey H. S. "Some theoretical elucidation of Cox
% reduction and confidence sets of models".


seed=11;
rng(seed)

p=1000;                     %number of covariates
n=100;                      %sample size
sVec=[5,3];                 %sparsity of unknown parameter in linear model
cVec=[5,3];                 %number of noise variables correlated with signal variables
rhoVec=[0.9,0.5];           %correlation
sigStrengthVec=[0.6,1];     %signal strength
sigma=1;                    %variance of noise in linear model
var=1;                      %variance of covariates
intercept=1;                %intercept term in linear model
propR1=0.4;                 %sample split: proportion of observations used in one portion of sample
alpha=0.01;                 %significance level for construction of the model confidence set
L2Start=0.05;               %Initial significance level in round 2 of Cox reduction 
L2StartRR=0.05;             %Initial significance level in round 2 of Cox reduction with re-randomisation
sizeUB=20;                  %maximum size of comprehensive model
sizeLB=10;
sizeCap=(sizeUB+sizeLB)/2;  %number of variables included in the comprehensive model by marginal screening
R=500;                      %number of Monte Carlo repetitions

numRows=9;
for s=sVec
    for c=cVec
        for rho=rhoVec
            for sigStrength=sigStrengthVec
                
                %Generate covariance matrix Sigma 
                cov1=rho.*ones(c+s)+(1-rho).*eye(c+s);
                covMatrixInit=[cov1, zeros(s+c,p-(s+c));zeros(p-(s+c),s+c), eye(p-(s+c))];
                covMatrix=diag(sqrt(var).*ones(p,1))*covMatrixInit*diag(sqrt(var).*ones(p,1));
                
                %Define unknown parameter in linear model
                trueBetaInit=[sigStrength.*ones(s,1);zeros(p-s,1)]; 
                
                %Matrices to store results
                coxReductionOriginal=NaN(numRows, R);       %Cox Reduction
                coxReductionSampleSplit=NaN(numRows,R);     %Cox Reduction with sample splitting
                coxReductionRandomised=NaN(numRows,R);      %Cox Reduction with re-randomisation
                coxReductionRandomisedSS=NaN(numRows,R);    %Cox Reduction with re-randomisation and sample splitting
                SIS=NaN(numRows,R);                         %Marginal screening
                undertunedLasso=NaN(numRows,R);             %Undertuned LASSO
                
                for r=1:R
                    r

                    %Generate design matrix
                    permuteVec=randperm(p);
                    trueBeta=trueBetaInit(permuteVec); 
                    I=eye(p);
                    permMatrix=I(permuteVec,:);
                    covPerm=permMatrix*covMatrix*(inv(permMatrix)); 
                    XAll=mvnrnd(zeros(p,1),covPerm,n); 
                    XAllScaled=normalize(XAll); 

                    %Generate response variable
                    epsilon=sqrt(sigma).*randn(n,1);
                    YAll=intercept.*ones(n,1)+XAll*trueBeta+epsilon;

                    %Randomly split sample in two
                    permuteObs=randperm(n);
                    obsR1=permuteObs(1:(propR1*n));
                    obsR2=permuteObs((propR1*n+1):end);
                    YSmall=YAll(obsR1);
                    YBig=YAll(obsR2);
                    XSmall=XAll(obsR1,:);
                    XBig=XAll(obsR2,:);
                    XSmallScaled=XAllScaled(obsR1,:);
                    XBigScaled=XAllScaled(obsR2,:);
                    
                    activeSet=find(trueBeta);                   %indices of signal variables
                    [rowIdx, ~]=find(covPerm==rho);         
                    corrSet=unique(rowIdx);                     %indices of variables correlated with the response
                    corrNoiseSet=setdiff(corrSet, activeSet);   %indices of noise variables correlated with the response
                    %noiseSet=find(~trueBeta);
                                        
                    %%%%%  Marginal screening %%%%%
                    [sortedCorr, sortedIdx]=sort(abs(corr(YBig, XBig)), 'descend');                                                              %sort covariates by their marginal correlation with the response
                    setSelectedMS=sortedIdx(1:sizeCap)';                                                                                         %set the comprehensive model as the first sizeCap variables
                    [fNoRejectSim,numberFalseModelsGoodFitFSim]=fTestModelsv2(YSmall,XSmall,setSelectedMS,alpha,activeSet); %construct model confidence set
                    isInCompMod=(all(ismember(activeSet,setSelectedMS))==1);                                                                     %check whether S (true model) is a subset of comprehensive model
                    SIS(4:end,r)=[mean(ismember(activeSet, setSelectedMS));mean(ismember(corrNoiseSet, setSelectedMS)); sizeCap; isInCompMod;fNoRejectSim;numberFalseModelsGoodFitFSim]; %record properties of model confidence set 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %%%%%  Undertuned Lasso  %%%%%
                    [lassoInfo,~]=lasso( XBigScaled,YBig);               %perform lasso regression at various values of tuning parameter
                    lambdaIdx=min(find(sum(abs(lassoInfo)>0)<=sizeCap)); %find index of smallest tuning parameter that produces a model with at most "sizeCap" variables
                    setSelectedLasso= find(lassoInfo(:,lambdaIdx));      %construct comprehensive model at this tuning parameter value
                    numSelectedLasso=size(setSelectedLasso,1);           %record size of comprehensive model
                    [fNoRejectSim,numberFalseModelsGoodFitFSim]=fTestModelsv2(YSmall,XSmall,setSelectedLasso,alpha,activeSet); % construct model confidence set
                    isInCompMod=(all(ismember(activeSet,setSelectedLasso))==1);
                    undertunedLasso(4:end,r)=[mean(ismember(activeSet, setSelectedLasso));mean(ismember(corrNoiseSet, setSelectedLasso)); numSelectedLasso;isInCompMod;fNoRejectSim;numberFalseModelsGoodFitFSim];
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    %%%%%  Original Cox Reduction  %%%%%
                    [setSelectedCube,numSelectedCube] = coxReductionR1(YBig,XBig);                                                                  %Round 1 of Cox Reduction using larger portion of sample
                    [setSelectedSquare, numSelectedSquare,L2Original]=coxReductionR2(YBig,XBig,L2Start,0.001, setSelectedCube, sizeCap);            %Round 2 of Cox Reduction using larger portion of sample   
                    [fNoRejectSim,numberFalseModelsGoodFitFSim]=fTestModelsv2(YSmall,XSmall,setSelectedSquare,alpha,activeSet);  %Construct confidence set of models using smaller portion of sample
                    isInCompMod=(all(ismember(activeSet,setSelectedSquare))==1);                    
                    coxReductionOriginal(:,r)=[mean(ismember(activeSet, setSelectedCube));mean(ismember(corrNoiseSet, setSelectedCube));numSelectedCube;mean(ismember(activeSet, setSelectedSquare));mean(ismember(corrNoiseSet, setSelectedSquare)); numSelectedSquare;isInCompMod;fNoRejectSim;numberFalseModelsGoodFitFSim];
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    %%%%%%  Sample splitting Cox reduction  %%%%%
                    [setSelectedCube,numSelectedCube] = coxReductionR1(YSmall,XSmall);                                                              %Round 1 of Cox Reduction using smaller portion of sample
                    [setSelectedSquare, numSelectedSquare,~]=coxReductionR2(YBig,XBig,L2Start,0.001, setSelectedCube, sizeCap);                     %Round 2 of Cox Reduction using larger portion of sample
                    [fNoRejectSim,numberFalseModelsGoodFitFSim]=fTestModelsv2(YSmall,XSmall,setSelectedSquare,alpha,activeSet);  %Construct confidence set of models using smaller portion of sample
                    isInCompMod=(all(ismember(activeSet,setSelectedSquare))==1);                    
                    coxReductionSampleSplit(:,r)=[mean(ismember(activeSet, setSelectedCube));mean(ismember(corrNoiseSet, setSelectedCube));numSelectedCube;mean(ismember(activeSet, setSelectedSquare));mean(ismember(corrNoiseSet, setSelectedSquare)); numSelectedSquare;isInCompMod;fNoRejectSim;numberFalseModelsGoodFitFSim];
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                     
                    %%%%%%   Re-randomised Cox reduction (with and without sample splitting) %%%%%%   
                    numRerandomisations=10; %number of randomisations of variable indiced in hypercube
                    countVec=zeros(p,1);    %vector recording the number of times each variable is retained (without sample splitting)
                    countVecSS=countVec;    %vector recording the number of times each variable is retained (with sample splitting)
                    for m=1:numRerandomisations
                        %%%%%%   Without sample splitting   %%%%%%   
                        [setSelectedCube,numSelectedCube] = coxReductionR1(YBig,XBig);                                                      %Round 1 of Cox Reduction using large portion of data
                        [setSelectedSquare, numSelectedSquare,~]=coxReductionR2(YBig,XBig,L2StartRR,0.001, setSelectedCube, 1.7*sizeCap);   %Round 2 of Cox Reduction using large portion of data
                        %Count the number of times each variable is retained
                        yes=zeros(p,1);
                        yes(setSelectedSquare,1)=1;
                        countVec=countVec+yes;
                        
                        %%%%%%   With sample splitting   %%%%%%  
                        [setSelectedCubeSS,numSelectedCubeSS] = coxReductionR1(YSmall,XSmall);                                                      %Round 1 of Cox Reduction using smaller portion of data
                        [setSelectedSquareSS, numSelectedSquareSS,~]=coxReductionR2(YBig,XBig,L2StartRR,0.001, setSelectedCubeSS, 1.7*sizeCap);     %Round 2 of Cox Reduction using larger portion of data
                        %Count the number of times each variable is retained
                        yesSS=zeros(p,1);
                        yesSS(setSelectedSquareSS,1)=1;
                        countVecSS=countVecSS+yesSS;
                    end
                    % Construct comprehensive model and confidence set of models (no sample splitting)
                    setSelectedB=find((countVec./numRerandomisations)>0.4); %comprehensive model = indices selected in at least 40% of re-randomisations
                    numSelectedB=length(setSelectedB);
                    [fNoRejectSim,numberFalseModelsGoodFitFSim]=fTestModelsv2(YSmall,XSmall,setSelectedB,alpha,activeSet); %construct model confidence set
                    isInCompMod=(all(ismember(activeSet,setSelectedB))==1);                    
                    coxReductionRandomised(4:end,r)=[mean(ismember(activeSet, setSelectedB));mean(ismember(corrNoiseSet, setSelectedB)); numSelectedB;isInCompMod;fNoRejectSim;numberFalseModelsGoodFitFSim];
                
                    % Construct comprehensive model and confidence set of models (sample splitting)
                    setSelectedBSS=find((countVecSS./numRerandomisations)>0.4); %comprehensive model = indices selected in at least 40% of re-randomisations
                    numSelectedBSS=length(setSelectedBSS);
                    [fNoRejectSim,numberFalseModelsGoodFitFSim]=fTestModelsv2(YSmall,XSmall,setSelectedBSS,alpha,activeSet); %construct model confidence set
                    isInCompMod=(all(ismember(activeSet,setSelectedBSS))==1);                    
                    coxReductionRandomisedSS(4:end,r)=[mean(ismember(activeSet, setSelectedBSS));mean(ismember(corrNoiseSet, setSelectedBSS)); numSelectedBSS;isInCompMod;fNoRejectSim;numberFalseModelsGoodFitFSim];
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                   
                end

                %%%%  Record and save results %%%%%%%%%%%%%%%
                colNames={'SIS', 'Lasso', 'CR: Original', 'CR: Sample Split', 'CR: Rerandomised', 'CR: Sample Split + Rerandomised '}; 
                rowNames={'R1: Proportion of signal retained', 'R1: Proportion of corr. noise retained', 'R1: Size of selected set', 'Comp. Model: Prop. signal retained', 'Comp. Model: Prop. of corr. noise retained', 'Size of comp. model','S in comp Model', 'S in M', 'Size of M\S'};
                results=[mean(SIS,2), mean(undertunedLasso,2), mean(coxReductionOriginal,2),  mean(coxReductionSampleSplit,2),mean(coxReductionRandomised,2),mean(coxReductionRandomisedSS,2)];
                array2table(results,'RowNames',rowNames,'VariableNames',colNames)
                
                colNames={'SIS', 'Lasso', 'CR: Original', 'CR: Sample Split', 'CR: Rerandomised', 'CR: Sample Split + Rerandomised '}; 
                rowNames={'R1: Proportion of signal retained', 'R1: Proportion of corr. noise retained', 'R1: Size of selected set', 'Comp. Model: Prop. signal retained', 'Comp. Model: Prop. of corr. noise retained', 'Size of comp. model','S in comp Model', 'S in M', 'Size of M\S'};
                results=[std(SIS,[],2), std(undertunedLasso,[],2), std(coxReductionOriginal,[],2),  std(coxReductionSampleSplit,[],2),std(coxReductionRandomised,[],2),std(coxReductionRandomisedSS,[],2)];
                array2table(results,'RowNames',rowNames,'VariableNames',colNames)
                
%               %mClock=ceil(clock);
%               %tStamp=1e6*mClock(2)+1e4*mClock(3)+100*mClock(4)+mClock(5);
                %fnme=['simuConfSets_n' num2str(n) '_alpha' num2str(alpha) '_s' num2str(s) '_sigToNoise' num2str(sigStrength/sigma) '_c' num2str(c) '_rho' num2str(rho) '.mat'];
                %save(fnme); % and save the workspace
                            
            end
        end
    end
end