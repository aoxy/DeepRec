rm -f *.csv
unzip D1_0.csv.zip
unzip D1_1.csv.zip
unzip D1_2.csv.zip
unzip D1_3.csv.zip
# unzip D1_4.csv.zip
# unzip D1_5.csv.zip
# unzip D1_6.csv.zip
# unzip D1_7.csv.zip
cat D1_*.csv > train.csv
rm -f D1_*.csv

unzip D8_0.csv.zip
unzip D8_1.csv.zip
# unzip D8_2.csv.zip
# unzip D8_3.csv.zip
cat D8_*.csv > eval.csv
rm -f D8_*.csv