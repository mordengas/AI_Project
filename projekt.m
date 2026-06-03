clear all;
clc;
data = readtable("train.csv",VariableNamingRule="preserve");

% Usunięcie kolumny ID (jeśli nie jest potrzebna)
data.Loan_ID = [];

for i = 1:width(data)
    col = data{:, i};
    missingRatio = sum(ismissing(col)) / height(data);  % procent braków w kolumnie
    
    if missingRatio > 0.05
        % Uzupełnij braki
        if isnumeric(col) || islogical(col)
            data{ismissing(col), i} = median(col, 'omitnan');
        elseif iscell(col)
            modeVal = mode(categorical(col));
            data(ismissing(col), i) = {char(modeVal)};
        else
            modeVal = mode(categorical(string(col)));
            data{ismissing(col), i} = char(modeVal);
        end
    else
        % Usuń wiersze z brakami w tej kolumnie
        data = rmmissing(data, 'DataVariables', data.Properties.VariableNames{i});
    end
end

% Gender: Female = 1, Male = 0
data.Gender = string(data.Gender);
data.Gender(data.Gender == "Female") = "1";
data.Gender(data.Gender == "Male") = "0";
data.Gender = str2double(data.Gender);

% Married: Yes = 1, No = 0
data.Married = string(data.Married);
data.Married(data.Married == "Yes") = "1";
data.Married(data.Married == "No")  = "0";
data.Married = str2double(data.Married);

% Dependents: '0', '1', '2', '3+'
data.Dependents = string(data.Dependents);
data.Dependents(data.Dependents == "3+") = "3";
data.Dependents = str2double(data.Dependents);

% Education: Graduate = 1, Not Graduate = 0
data.Education = string(data.Education);
data.Education(data.Education == "Graduate") = "1";
data.Education(data.Education == "Not Graduate") = "0";
data.Education = str2double(data.Education);

% Self_Employed: Yes = 1, No = 0
data.Self_Employed = string(data.Self_Employed);
data.Self_Employed(data.Self_Employed == "Yes") = "1";
data.Self_Employed(data.Self_Employed == "No")  = "0";
data.Self_Employed = str2double(data.Self_Employed);

% Property_Area: Rural = 0, Semiurban = 1, Urban = 2
data.Property_Area = string(data.Property_Area);
data.Property_Area(data.Property_Area == "Rural")     = "0";
data.Property_Area(data.Property_Area == "Semiurban") = "1";
data.Property_Area(data.Property_Area == "Urban")     = "2";
data.Property_Area = str2double(data.Property_Area);

% Loan_Status: N = 0, Y = 1
data.Loan_Status = string(data.Loan_Status);
data.Loan_Status(data.Loan_Status == "N") = "0";
data.Loan_Status(data.Loan_Status == "Y") = "1";
data.Loan_Status = str2double(data.Loan_Status);

% Lista kolumn do normalizacji
colsToNormalize = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"];

for i = 1:length(colsToNormalize)
    colName = colsToNormalize(i);
    colData = data.(colName);

    % Upewnij się, że kolumna nie ma NaN
    if any(ismissing(colData))
        warning("Kolumna %s nadal zawiera braki!", colName);
    end

    % Normalizacja min-max
    colMin = min(colData);
    colMax = max(colData);

    if colMax ~= colMin
        data.(colName) = (colData - colMin) / (colMax - colMin);
    else
        data.(colName) = zeros(size(colData));  
    end
end

% Zapisz do pliku CSV
% writetable(data, 'cleaned_normalised_data.csv');

P = table2array(data(:, 1:end-1))';
T = table2array(data(:, end))';

net = newff(P, T, 30 ,{'logsig', 'purelin'});

net.trainParam.lr = 0.005;
net.trainParam.epochs = 20;
net.trainParam.goal = 0.0005;

net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;

[net, TR] = train(net, P, T);

Ptest = P(:, TR.testInd);
Ttest = T(:, TR.testInd);
Ytest = sim(net, Ptest);
Yclass = Ytest > 0.5;

Ttest = double(Ttest);
Yclass = double(Yclass);

cm = confusionmat(Ttest, Yclass);

TP = cm(2,2);
FP = cm(1,2);
FN = cm(2,1);
TN = cm(1,1);

confusionchart(cm);

accuracy = (TP + TN) / sum(cm(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);

fprintf('Accuracy: %.2f\n', accuracy*100);
fprintf('Precision: %.2f\n', precision*100);
fprintf('Recall: %.2f\n', recall*100);

