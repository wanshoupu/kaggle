import csv
with open('data.csv', 'a') as fp:
    for i in range(0,10):
        a = csv.writer(fp, delimiter=',');
        data = [[i,1,2,3],[i,2,3,4],[i,3,4,5,6]];
        a.writerows(data);
