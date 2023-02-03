function [fNoRejectSim,numberFalseModelsGoodFitFSim] = fTestModelsv2(Y,X,setSelected,alpha,activeSet)

% This function constructs a model confidence set based on a linear reponse,
% a design matrix and a comprehensive model.

% Input:
% - Y (the response vector): a vector
% - X (the design matrix without an intercept column): a matrix with the same 
%   number of rows as Y
% - setSelected (the comprehensive model): a vector with numbers indexing
%   columns of X
% - alpha (the significance level): a number between 0 and 1
% - activeSet (the set of signal variables): a vector indexing
%   the columns of X that correspond to signal variables.

% Output:
% - fNoRejectSim: takes the value 1 when the true model is contained in the 
%   confidence set and 0 otherwise,
% - numberFalseModelsGoodFitFSim: the number of false models included in 
%   the model confidence set,


fNoRejectSim=0;                         %takes the value 1 when the true model is contained in the confidence set and 0 otherwise
s=size(activeSet,1);                    %number of signal variables
modelSize=min(5,size(setSelected,1));   %maximum size of models included in the confidence set
[n,~]=size(X);
countModelsF=0;                         %counts the number of models included in the confidence set

% Compute the residuals and degrees of freedom for a regression of Y on the
% covariates indexed by the comprehensive model.
[~,~,residBig]=regress(Y,[ones(n,1), X(:,setSelected)]);
RSSBig=residBig'*residBig;
dfBig=n-(size(setSelected,1)+1);

%Iterate over all sub-models of the comprehensive model by their size
    for j=1:modelSize
        combinationMatrix=nchoosek(setSelected',j);          %identify all sub-models of the comprehensive model of size j
        logicFitVectorF=zeros(size(combinationMatrix,1),1);  %vector with entries equal to 1 when a model is included in the confidence set
        for l=1:size(combinationMatrix,1)
           %compute residuals and degrees of freedom for sub-model
           XSelect=X(:,combinationMatrix(l,:));
           [beta,~,residSmall]=regress(Y,[ones(n,1), XSelect]);
           RSSSmall=residSmall'*residSmall;
           diffRSS=RSSSmall-RSSBig;
           dfSmall=n-(size(XSelect,2)+1);

           %compute the critical value of the test
           fCrit=finv(1-alpha,dfSmall-dfBig,dfBig);

           %perform the test
           if isnan(fCrit) %occurs when df=0
              logicFitVectorF(l,1)=1;
           else
                if (diffRSS/(dfSmall-dfBig))/(RSSBig/dfBig)<=fCrit
                    logicFitVectorF(l,1)=1;
                end
           end

           %check whether the true set of signal variables is included in the confidence set of models
           if ((j==s)&&(all(ismember(activeSet,combinationMatrix(l,:)))))
              fNoRejectSim=logicFitVectorF(l,1);
           end
        end
        %count the number of models included in the model confidence set
        countModelsF=countModelsF+sum(logicFitVectorF,1);
    end
    %count the number of false models included in the model confidence set
    if fNoRejectSim>0
        numberFalseModelsGoodFitFSim=countModelsF-fNoRejectSim; 
    else
        numberFalseModelsGoodFitFSim=countModelsF;
    end
end


